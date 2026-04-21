"""Output TUM directory format with rich metadata."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from .camera import CameraIntrinsics, CameraExtrinsics, DepthFormat, SensorConfig
from .trajectory import Trajectory


class TUMFormatter:
    """Format converted bag into TUM RGB-D directory structure."""
    
    def __init__(
        self,
        output_dir: Path,
        bag_name: str,
        intrinsics: Optional[CameraIntrinsics] = None,
        depth_format: Optional[DepthFormat] = None,
        sensor_config: Optional[SensorConfig] = None,
    ):
        self.output_dir = output_dir / bag_name
        self.rgb_dir = self.output_dir / "rgb"
        self.depth_dir = self.output_dir / "depth"
        
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        
        self.intrinsics = intrinsics
        self.depth_format = depth_format
        self.sensor_config = sensor_config
        
        self._rgb_timestamps: list[tuple[float, Path]] = []
        self._depth_timestamps: list[tuple[float, Path]] = []
    
    def save_rgb_image(self, timestamp_ns: int, img_data: bytes, fmt: str) -> Path:
        """Save RGB image and record timestamp."""
        fmt_lower = fmt.lower().strip()

        # Pre-encoded PNG - write directly
        if fmt_lower == "png_preencoded":
            ts_sec = timestamp_ns / 1e9
            filename = f"{ts_sec:.6f}.png"
            filepath = self.rgb_dir / filename
            with open(filepath, 'wb') as f:
                f.write(img_data)
            self._rgb_timestamps.append((ts_sec, filepath))
            return filepath

        if "jpeg" in fmt_lower or "jpg" in fmt_lower:
            arr = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif "png" in fmt_lower:
            arr = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        elif "bgr8" in fmt_lower or "rgb8" in fmt_lower:
            arr = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            # Try raw bytes as image
            arr = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)  # fallback

        if img is None:
            raise ValueError(f"Failed to decode image format {fmt}")

        # Convert to PNG for compatibility
        ts_sec = timestamp_ns / 1e9
        filename = f"{ts_sec:.6f}.png"
        filepath = self.rgb_dir / filename
        cv2.imwrite(str(filepath), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])

        self._rgb_timestamps.append((ts_sec, filepath))
        return filepath
    
    def save_depth_image(self, timestamp_ns: int, depth_data: bytes, fmt: str) -> Path:
        """Save depth image as 16-bit PNG preserving mm precision."""
        fmt_lower = fmt.lower().strip()

        # Pre-encoded PNG - write directly
        if fmt_lower == "png_preencoded":
            ts_sec = timestamp_ns / 1e9
            filename = f"{ts_sec:.6f}.png"
            filepath = self.depth_dir / filename
            with open(filepath, 'wb') as f:
                f.write(depth_data)
            self._depth_timestamps.append((ts_sec, filepath))
            return filepath

        if "png" in fmt_lower or "compresseddepth" in fmt_lower:
            # PNG compressed depth (RealSense, Kinect, etc.)
            arr = np.frombuffer(depth_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        elif "jpeg" in fmt_lower:
            arr = np.frombuffer(depth_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        else:
            # Try raw 16-bit
            try:
                arr = np.frombuffer(depth_data, dtype=np.uint16)
                img = arr.reshape(-1)
            except Exception:
                arr = np.frombuffer(depth_data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Failed to decode depth format {fmt}")

        ts_sec = timestamp_ns / 1e9
        filename = f"{ts_sec:.6f}.png"
        filepath = self.depth_dir / filename

        # Ensure 16-bit
        if img.dtype != np.uint16:
            img = img.astype(np.uint16)

        cv2.imwrite(str(filepath), img)
        self._depth_timestamps.append((ts_sec, filepath))
        return filepath
    
    def save_timestamps(self):
        """Save rgb.txt and depth.txt timestamp files."""
        # Sort by timestamp
        self._rgb_timestamps.sort(key=lambda x: x[0])
        self._depth_timestamps.sort(key=lambda x: x[0])
        
        rgb_txt = self.output_dir / "rgb.txt"
        depth_txt = self.output_dir / "depth.txt"
        
        with open(rgb_txt, 'w') as f:
            f.write("# timestamp rgb/{filename}\n")
            for ts, path in self._rgb_timestamps:
                f.write(f"{ts:.6f} rgb/{path.name}\n")
        
        with open(depth_txt, 'w') as f:
            f.write("# timestamp depth/{filename}\n")
            for ts, path in self._depth_timestamps:
                f.write(f"{ts:.6f} depth/{path.name}\n")
    
    def save_metadata(self, extra: dict = None):
        """Save all metadata files."""
        import yaml
        
        # intrinsics.yaml
        if self.intrinsics:
            (self.output_dir / "intrinsics.yaml").write_text(
                yaml.dump(self.intrinsics.to_yaml(), default_flow_style=False)
            )
        
        # extrinsics.yaml (if available, saved separately)
        # depth_format.txt
        if self.depth_format:
            df_path = self.output_dir / "depth_format.txt"
            with open(df_path, 'w') as f:
                for key, val in self.depth_format.to_yaml().items():
                    f.write(f"{key}: {val}\n")
        
        # sensor.yaml
        if self.sensor_config:
            (self.output_dir / "sensor.yaml").write_text(
                yaml.dump(self.sensor_config.to_yaml(), default_flow_style=False)
            )
        
        # metadata.yaml - comprehensive bag info
        metadata = {
            "conversion": {
                "tool": "rosbag-to-tum",
                "version": "0.1.0",
            },
            "output_dirs": {
                "rgb": str(self.rgb_dir.relative_to(self.output_dir)),
                "depth": str(self.depth_dir.relative_to(self.output_dir)),
            },
            "topics": {
                "rgb": str(self._rgb_timestamps[0][1].name) if self._rgb_timestamps else "",
                "depth": str(self._depth_timestamps[0][1].name) if self._depth_timestamps else "",
            },
            "image_count": {
                "rgb": len(self._rgb_timestamps),
                "depth": len(self._depth_timestamps),
            }
        }
        
        if extra:
            metadata.update(extra)
        
        (self.output_dir / "metadata.yaml").write_text(
            yaml.dump(metadata, default_flow_style=False, sort_keys=False)
        )
    
    def finalize(self, trajectory: Trajectory | None = None, has_visual_fallback: bool = False):
        """Finalize output - save timestamps and trajectory files."""
        self.save_timestamps()
        
        if trajectory:
            trajectory.save_multiple(
                self.output_dir / "trajectory_file.txt",
                "groundtruth",
                "trajectory"
            )
        
        self.save_metadata({
            "has_trajectory": trajectory is not None,
            "trajectory_source": trajectory.source if trajectory else None,
            "has_visual_fallback": has_visual_fallback,
        })