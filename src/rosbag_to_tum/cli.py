"""CLI for rosbag-to-tum conversion tool."""

from __future__ import annotations

import math
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import click
from rich.console import Console
from rich.progress import Progress, TaskID, track
from rich.table import Table

from .reader import MCAPBagReader, detect_image_topics, detect_camera_info_topics, parse_compressed_image_message
from .camera import CameraIntrinsics, DepthFormat, SensorConfig, parse_camera_info_message
from .trajectory import extract_trajectory_from_tf_odom, Trajectory, TrajectoryEntry
from .visual_tracker import VisualTracker, estimate_trajectory_from_files
from .formatter import TUMFormatter
from .metadata import BagMetadata, analyze_bag, extract_camera_params, create_sensor_config


console = Console()


@dataclass
class BagData:
    """Single-pass collected bag data."""
    topics: dict = field(default_factory=dict)
    rgb_cam_info: dict | None = None
    depth_cam_info: dict | None = None
    rgb_frames: list = field(default_factory=list)  # [(ts_ns, raw_data)]
    depth_frames: list = field(default_factory=list)
    odom_entries: list[TrajectoryEntry] = field(default_factory=list)
    tf_entries: list[TrajectoryEntry] = field(default_factory=list)
    rgb_topic: str = ""
    depth_topic: str = ""
    aligned_depth_topic: str = ""
    has_odom: bool = False
    has_tf: bool = False
    has_imu: bool = False


def find_mcap_bags(root_path: Path) -> list[tuple[Path, Path]]:
    """Find all .mcap files and their parent directories (bag folders)."""
    bags = []
    for mcap in root_path.rglob("*.mcap"):
        bag_dir = mcap.parent
        bags.append((bag_dir, mcap))

    # Deduplicate by bag dir
    seen = set()
    unique = []
    for bag_dir, mcap_path in bags:
        if bag_dir not in seen:
            seen.add(bag_dir)
            unique.append((bag_dir, mcap_path))

    return unique


def check_already_converted(output_dir: Path, bag_name: str) -> bool:
    """Check if bag already has complete conversion."""
    out = output_dir / bag_name
    if not out.exists():
        return False

    # Check for essential files
    required = ["rgb.txt", "depth.txt", "metadata.yaml"]
    for f in required:
        if not (out / f).exists():
            return False

    # Check image count
    rgb_dir = out / "rgb"
    depth_dir = out / "depth"
    if rgb_dir.exists() and depth_dir.exists():
        rgb_count = len(list(rgb_dir.glob("*.png")))
        depth_count = len(list(depth_dir.glob("*.png")))
        return rgb_count > 0 and depth_count > 0

    return False


def _encode_image(args: tuple) -> tuple[int, bytes, str] | None:
    """Encode a single image to PNG. Worker function for parallel processing.

    Returns (timestamp_ns, png_bytes, "rgb"/"depth") or None on failure.
    """
    import cv2
    import numpy as np
    from .reader import parse_compressed_image_message

    ts, raw_data, image_type = args

    try:
        _, _, fmt, img_raw = parse_compressed_image_message(raw_data)

        if fmt not in ("jpeg", "png"):
            return None

        if fmt == "jpeg":
            img_arr = np.frombuffer(img_raw, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        else:
            img_arr = np.frombuffer(img_raw, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
            if img is not None and len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img is None:
            return None

        # Encode as PNG
        success, encoded = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if success:
            return (ts, encoded.tobytes(), image_type)
    except Exception:
        pass
    return None


def _has_valid_tf_entries(tf_entries: list) -> bool:
    """Check if TF entries contain actual usable trajectory (not zeros or garbage)."""
    if not tf_entries or len(tf_entries) < 10:
        return False

    # Check for valid values: position magnitude < 100m and valid quaternion
    valid_count = 0
    for e in tf_entries:
        # Reject if position is exactly zero (likely uninitialized)
        pos_zero = (e.tx == 0.0 and e.ty == 0.0 and e.tz == 0.0)
        # Reject if quaternion is exactly zero (invalid)
        quat_zero = (e.qx == 0.0 and e.qy == 0.0 and e.qz == 0.0 and e.qw == 0.0)
        # Reject if position magnitude is impossibly large (> 100m)
        pos_too_large = (abs(e.tx) > 100 or abs(e.ty) > 100 or abs(e.tz) > 100)
        # Reject if position magnitude is too small (garbage that looks like uninitialized)
        pos_too_small = (abs(e.tx) < 1e-6 and abs(e.ty) < 1e-6 and abs(e.tz) < 1e-6)
        # Reject if quaternion has NaN or Inf
        quat_invalid = (math.isnan(e.qw) or math.isnan(e.qx) or
                       math.isnan(e.qy) or math.isnan(e.qz) or
                       math.isinf(e.qw) or math.isinf(e.qx) or
                       math.isinf(e.qy) or math.isinf(e.qz))

        if not pos_zero and not quat_zero and not pos_too_large and not pos_too_small and not quat_invalid:
            valid_count += 1

    # Require at least 10% of entries to be valid
    return valid_count > len(tf_entries) * 0.1


def collect_bag_data_single_pass(
    reader,
    max_frames: int | None,
    visual_tracker_only: bool,
) -> tuple[BagData, dict]:
    """Collect all bag data in a SINGLE pass through the bag.

    Returns (bag_data, topics).
    """
    from .trajectory import parse_tf_message, parse_odometry_message

    data = BagData()
    topics: dict[str, Any] = {}

    frame_count = 0
    rgb_topic = None
    depth_topic = None
    aligned_depth_topic = None
    rgb_cam_info_topic = None
    depth_cam_info_topic = None

    for topic, ts, msg_data in reader.iter_messages():
        # Track topics
        if topic not in topics:
            topics[topic] = {"count": 0, "type": ""}
        topics[topic]["count"] += 1

        t_lower = topic.lower()

        # Detect topic roles
        if "camera_info" in t_lower:
            if "color" in t_lower and not rgb_cam_info_topic:
                rgb_cam_info_topic = topic
                data.rgb_cam_info = parse_camera_info_message(msg_data)
            elif ("depth" in t_lower or "aligned" in t_lower) and not depth_cam_info_topic:
                depth_cam_info_topic = topic
                data.depth_cam_info = parse_camera_info_message(msg_data)

        # Detect image topics - check aligned BEFORE color to avoid misclassification
        if "aligned" in t_lower and "depth" in t_lower and ("image" in t_lower or "compressed" in t_lower):
            if not aligned_depth_topic:
                aligned_depth_topic = topic
                data.aligned_depth_topic = topic
        elif "color" in t_lower and ("image" in t_lower or "compressed" in t_lower):
            if not rgb_topic:
                rgb_topic = topic
                data.rgb_topic = topic
        elif "depth" in t_lower and "aligned" not in t_lower and ("image" in t_lower or "compressed" in t_lower):
            if not depth_topic:
                depth_topic = topic
                data.depth_topic = topic

        # Check for odom/TF/IMU
        if "odom" in t_lower and not data.has_odom:
            data.has_odom = True
        if "tf" in t_lower and "static" not in t_lower and not data.has_tf:
            data.has_tf = True
        if "imu" in t_lower and not data.has_imu:
            data.has_imu = True

        # Collect RGB frames
        if rgb_topic and topic == rgb_topic:
            data.rgb_frames.append((ts, msg_data))
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break

        # Collect depth frames
        depth_topic_to_use = depth_topic or aligned_depth_topic
        if depth_topic_to_use and topic == depth_topic_to_use:
            data.depth_frames.append((ts, msg_data))

        # Collect odometry
        if not visual_tracker_only and data.has_odom and "odom" in t_lower:
            result = parse_odometry_message(msg_data)
            if result:
                ts_odom, (tx, ty, tz), (qw, qx, qy, qz) = result
                data.odom_entries.append(TrajectoryEntry(
                    timestamp_ns=ts_odom, tx=tx, ty=ty, tz=tz,
                    qw=qw, qx=qx, qy=qy, qz=qz, source="odom"
                ))

        # Collect TF
        if not visual_tracker_only and data.has_tf and topic == "/tf":
            transforms = parse_tf_message(msg_data)
            for parent, child, ts_tf, xyz, quat in transforms:
                if "camera" in child or "base" in child or "odom" in child:
                    tx, ty, tz = xyz
                    qw, qx, qy, qz = quat
                    data.tf_entries.append(TrajectoryEntry(
                        timestamp_ns=ts_tf, tx=tx, ty=ty, tz=tz,
                        qw=qw, qx=qx, qy=qy, qz=qz, source="tf"
                    ))

    return data, topics


def process_images_parallel(
    rgb_frames: list,
    depth_frames: list,
    formatter: TUMFormatter,
    workers: int | None,
    console: Console,
) -> int:
    """Process images in parallel using worker pool.

    Returns number of RGB frames extracted.
    """
    if workers is None:
        workers = os.cpu_count() or 4

    # Prepare all frames with type
    all_frames = [(ts, data, "rgb") for ts, data in rgb_frames]
    all_frames.extend((ts, data, "depth") for ts, data in depth_frames)

    console.print(f"  Encoding {len(rgb_frames)} RGB, {len(depth_frames)} depth frames with {workers} workers...")

    encoded_frames = []

    if workers > 1 and len(all_frames) > 10:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_encode_image, f) for f in all_frames]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    encoded_frames.append(result)
    else:
        for f in all_frames:
            result = _encode_image(f)
            if result:
                encoded_frames.append(result)

    # Save frames
    for ts, png_bytes, img_type in encoded_frames:
        if img_type == "rgb":
            formatter.save_rgb_image(ts, png_bytes, "png_preencoded")
        else:
            formatter.save_depth_image(ts, png_bytes, "png_preencoded")

    rgb_count = sum(1 for _, _, t in encoded_frames if t == "rgb")
    return rgb_count


@click.command()
@click.argument("bags_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--skip-existing", is_flag=True, help="Skip bags already converted")
@click.option("--visual-tracker", is_flag=True, help="Force visual tracker even if odometry available")
@click.option("--visual-tracker-only", is_flag=True, help="Only use visual tracker (no odometry/TF)")
@click.option("--max-frames", type=int, default=None, help="Max frames per bag (for testing)")
@click.option("--bag-filter", type=str, default=None, help="Only process bags matching this name pattern")
@click.option("--workers", "-j", type=int, default=None, help="Number of parallel workers (default: all CPU cores)")
def main(
    bags_path: Path,
    output_dir: Path,
    skip_existing: bool,
    visual_tracker: bool,
    visual_tracker_only: bool,
    max_frames: int | None,
    bag_filter: str | None,
    workers: int | None,
):
    """Convert ROS bags (MCAP format) to TUM RGB-D dataset format.

    BAGS_PATH: Root directory containing bag folders (each with .mcap file)
    OUTPUT_DIR: Directory where TUM format datasets will be written
    """
    console.print(f"\n[bold cyan]rosbag-to-tum[/bold cyan] - ROS bag to TUM RGB-D converter")
    console.print(f"Bags path: {bags_path}")
    console.print(f"Output dir: {output_dir}\n")

    # Find all bags
    bags = find_mcap_bags(bags_path)
    console.print(f"Found {len(bags)} bag(s)\n")

    results = []

    for bag_dir, mcap_path in track(list(bags), description="Processing bags"):
        bag_name = bag_dir.name

        if bag_filter and bag_filter not in bag_name:
            continue

        console.print(f"\n[bold]Processing: {bag_name}[/bold]")
        console.print(f"  MCAP: {mcap_path}")

        if skip_existing and check_already_converted(output_dir, bag_name):
            console.print(f"  [yellow]Skipping (already converted)[/yellow]")
            results.append((bag_name, "skipped"))
            continue

        try:
            with MCAPBagReader(mcap_path) as reader:
                # SINGLE PASS: collect all data
                bag_data, topics = collect_bag_data_single_pass(
                    reader, max_frames, visual_tracker_only
                )

                console.print(f"  Topics: {len(topics)}")
                for t, info in topics.items():
                    console.print(f"    - {t} ({info.get('count', 0)} msgs)")

                # Build metadata
                metadata = BagMetadata()
                metadata.bag_name = bag_name
                metadata.bag_path = str(mcap_path)
                metadata.topics = {t: info.get("type", "") for t, info in topics.items()}
                metadata.rgb_topic = bag_data.rgb_topic
                metadata.depth_topic = bag_data.depth_topic
                metadata.aligned_depth_topic = bag_data.aligned_depth_topic
                metadata.has_odom = bag_data.has_odom
                metadata.has_tf = bag_data.has_tf
                metadata.has_imu = bag_data.has_imu

                # Build intrinsics from collected CameraInfo
                if bag_data.rgb_cam_info:
                    ci = bag_data.rgb_cam_info
                    metadata.intrinsics_rgb = CameraIntrinsics(
                        model=ci.model,
                        width=ci.width,
                        height=ci.height,
                        fx=ci.fx,
                        fy=ci.fy,
                        cx=ci.cx,
                        cy=ci.cy,
                        k1=ci.k1,
                        k2=ci.k2,
                        p1=ci.p1,
                        p2=ci.p2,
                        k3=ci.k3,
                    )

                if bag_data.depth_cam_info:
                    ci = bag_data.depth_cam_info
                    metadata.intrinsics_depth = CameraIntrinsics(
                        model=ci.model,
                        width=ci.width,
                        height=ci.height,
                        fx=ci.fx,
                        fy=ci.fy,
                        cx=ci.cx,
                        cy=ci.cy,
                    )

                # Depth format - RealSense uses mm
                metadata.depth_format = DepthFormat(
                    unit="millimeters",
                    scale=1.0,
                    description="Depth in millimeters, scale=1.0 (NOT TUM's 5000 factor)"
                )

                # Sensor config - detect from topic names
                from .metadata import detect_sensor_type
                sensor_type = detect_sensor_type(topics)
                sensor_config = SensorConfig(sensor_type=sensor_type)
                metadata.sensor_config = sensor_config

                # Create formatter
                formatter = TUMFormatter(
                    output_dir=output_dir,
                    bag_name=bag_name,
                    intrinsics=metadata.intrinsics_rgb,
                    depth_format=metadata.depth_format,
                    sensor_config=sensor_config,
                )

                # Extract trajectory
                trajectory = None
                has_visual_fallback = False

                if visual_tracker_only:
                    console.print(f"  [blue]Using visual tracker only[/blue]")
                elif bag_data.odom_entries:
                    console.print(f"  [green]Extracting trajectory from odometry[/green]")
                    trajectory = Trajectory(entries=bag_data.odom_entries, source="odom")
                    trajectory.sort_by_time()
                elif bag_data.tf_entries and _has_valid_tf_entries(bag_data.tf_entries):
                    console.print(f"  [green]Extracting trajectory from TF[/green]")
                    trajectory = Trajectory(entries=bag_data.tf_entries, source="tf")
                    trajectory.sort_by_time()
                else:
                    console.print(f"  [yellow]No odometry/TF - will use visual tracker fallback[/yellow]")
                    has_visual_fallback = True

                # Process images in parallel
                if not bag_data.rgb_topic:
                    console.print(f"  [red]No RGB topic found, skipping[/red]")
                    results.append((bag_name, "error_no_rgb_topic"))
                    continue

                frame_count = process_images_parallel(
                    bag_data.rgb_frames, bag_data.depth_frames,
                    formatter, workers, console
                )

                console.print(f"  Extracted {frame_count} RGB frames")

                # Visual tracker if no trajectory (from already-extracted files)
                if trajectory is None and formatter.intrinsics is not None:
                    console.print(f"  Running visual tracker on extracted images...")
                    intrinsics = [
                        formatter.intrinsics.fx,
                        formatter.intrinsics.fy,
                        formatter.intrinsics.cx,
                        formatter.intrinsics.cy
                    ]
                    trajectory = estimate_trajectory_from_files(
                        formatter.rgb_dir, formatter.depth_dir, intrinsics, max_frames
                    )
                    if len(trajectory.entries) > 0:
                        console.print(f"  Visual tracker: {len(trajectory.entries)} poses")

                # Finalize
                formatter.finalize(trajectory, has_visual_fallback)

                console.print(f"  [green]Done! Output: {formatter.output_dir}[/green]")
                results.append((bag_name, "success"))

        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()
            results.append((bag_name, f"error: {e}"))

    # Summary
    console.print("\n[bold]Summary[/bold]")
    table = Table(show_header=True)
    table.add_column("Bag")
    table.add_column("Status")
    for name, status in results:
        color = "green" if status == "success" else "yellow" if status == "skipped" else "red"
        table.add_row(name, f"[{color}]{status}[/{color}]")
    console.print(table)


if __name__ == "__main__":
    main()
