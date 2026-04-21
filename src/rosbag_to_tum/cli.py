"""CLI for rosbag-to-tum conversion tool."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

import click
from rich.console import Console
from rich.progress import Progress, TaskID, track
from rich.table import Table

from .reader import MCAPBagReader, detect_image_topics, detect_camera_info_topics, parse_compressed_image
from .camera import CameraIntrinsics, DepthFormat, SensorConfig, parse_camera_info_message
from .trajectory import extract_trajectory_from_tf_odom
from .visual_tracker import VisualTracker, estimate_trajectory_from_images
from .formatter import TUMFormatter
from .metadata import BagMetadata, analyze_bag, extract_camera_params, create_sensor_config


console = Console()


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


def _process_images_parallel(
    reader,
    rgb_topic: str | None,
    depth_topic: str | None,
    formatter: TUMFormatter,
    max_frames: int | None,
    workers: int | None,
    console: Console,
) -> int:
    """Process images in parallel using worker pool.

    Returns number of RGB frames extracted.
    """
    import cv2
    import numpy as np

    if workers is None:
        workers = os.cpu_count() or 4

    # First pass: collect all image data
    rgb_frames: list[tuple[int, bytes, str]] = []  # (ts, raw_data, "rgb")
    depth_frames: list[tuple[int, bytes, str]] = []  # (ts, raw_data, "depth")
    frame_count = 0

    for topic, ts, data in reader.iter_messages():
        if rgb_topic and topic == rgb_topic:
            rgb_frames.append((ts, data, "rgb"))
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
        elif depth_topic and topic == depth_topic:
            depth_frames.append((ts, data, "depth"))

    console.print(f"  Collected {len(rgb_frames)} RGB, {len(depth_frames)} depth frames")

    # Parallel encoding
    all_frames = rgb_frames + depth_frames
    encoded_frames: list[tuple[int, bytes, str]] = []

    if workers > 1 and len(all_frames) > 10:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_encode_image, f) for f in all_frames]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    encoded_frames.append(result)
    else:
        # Sequential fallback
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


def _process_frame_batch(args: tuple) -> list[tuple[int, bytes, str]]:
    """Process a batch of frames in parallel (worker function).

    Returns list of (timestamp_ns, image_data, format, is_rgb) tuples.
    """
    import cv2
    import numpy as np
    from .reader import parse_compressed_image_message

    batch, = args  # tuple of batches
    results = []

    for topic, ts, data in batch:
        try:
            _, _, fmt, img_raw = parse_compressed_image_message(data)

            if fmt not in ("jpeg", "png"):
                continue

            if fmt == "jpeg":
                img_arr = np.frombuffer(img_raw, dtype=np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            else:
                img_arr = np.frombuffer(img_raw, dtype=np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
                if img is not None and len(img.shape) == 2:
                    # Convert grayscale to BGR for consistent handling
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if img is not None:
                # Encode as PNG
                success, encoded = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                if success:
                    results.append((ts, encoded.tobytes(), fmt))
        except Exception:
            continue

    return results
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
            # Read bag
            with MCAPBagReader(mcap_path) as reader:
                topics = reader.topics()
                
                # Analyze bag
                metadata = analyze_bag(reader, topics)
                metadata.bag_name = bag_name
                metadata.bag_path = str(mcap_path)
                
                console.print(f"  Topics: {len(topics)}")
                for t, info in topics.items():
                    console.print(f"    - {t} ({info.type}, {info.message_count} msgs)")
                
                # Extract camera params
                metadata = extract_camera_params(reader, metadata)
                
                # Create sensor config
                sensor_config = create_sensor_config(metadata, reader)
                
                # Create formatter
                formatter = TUMFormatter(
                    output_dir=output_dir,
                    bag_name=bag_name,
                    intrinsics=metadata.intrinsics_rgb,
                    depth_format=metadata.depth_format,
                    sensor_config=sensor_config,
                )
                
                # Extract or estimate trajectory
                trajectory = None
                has_visual_fallback = False
                
                if visual_tracker_only:
                    console.print(f"  [blue]Using visual tracker only[/blue]")
                elif metadata.has_odom or metadata.has_tf:
                    console.print(f"  [green]Extracting trajectory from odometry/TF[/green]")
                    trajectory = extract_trajectory_from_tf_odom(reader, list(topics.keys()))
                else:
                    console.print(f"  [yellow]No odometry/TF - will use visual tracker fallback[/yellow]")
                    has_visual_fallback = True
                
                # Process images in parallel
                # Prefer color topic for RGB, fallback to aligned_depth if needed
                rgb_topic = metadata.rgb_topic or metadata.aligned_depth_topic
                # Prefer depth topic, but aligned_depth works too
                depth_topic = metadata.depth_topic or metadata.aligned_depth_topic

                if not rgb_topic:
                    console.print(f"  [red]No RGB topic found, skipping[/red]")
                    results.append((bag_name, "error_no_rgb_topic"))
                    continue

                frame_count = _process_images_parallel(
                    reader, rgb_topic, depth_topic, formatter, max_frames, workers, console
                )

                console.print(f"  Extracted {frame_count} RGB frames")
                
                # Visual tracker if no trajectory
                if trajectory is None:
                    console.print(f"  Running visual tracker...")
                    if metadata.intrinsics_rgb:
                        K = [metadata.intrinsics_rgb.fx, metadata.intrinsics_rgb.fy,
                             metadata.intrinsics_rgb.cx, metadata.intrinsics_rgb.cy]
                    else:
                        K = [600.0, 600.0, 320.0, 240.0]  # defaults
                    
                    trajectory = estimate_trajectory_from_images(
                        reader, rgb_topic, depth_topic, K,
                        metadata.start_time_ns, metadata.end_time_ns
                    )
                
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