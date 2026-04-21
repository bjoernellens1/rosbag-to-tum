"""Extract and manage camera intrinsics, extrinsics, and depth format."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np
import yaml


@dataclass
class CameraIntrinsics:
    model: str = "pinhole"  # or "opencv" for full distortion model
    width: int = 0
    height: int = 0
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def to_yaml(self) -> dict:
        """Convert to plain dict with native Python types for YAML serialization."""
        return {
            'model': str(self.model),
            'width': int(self.width),
            'height': int(self.height),
            'fx': float(self.fx),
            'fy': float(self.fy),
            'cx': float(self.cx),
            'cy': float(self.cy),
            'k1': float(self.k1),
            'k2': float(self.k2),
            'p1': float(self.p1),
            'p2': float(self.p2),
            'k3': float(self.k3),
        }


@dataclass
class CameraExtrinsics:
    """Transform from camera optical frame to base/body frame."""
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w, x, y, z

    def to_yaml(self) -> dict:
        return asdict(self)


@dataclass
class DepthFormat:
    unit: str = "millimeters"  # "millimeters", "meters", "meters_1000"
    scale: float = 1.0
    description: str = ""

    def to_yaml(self) -> dict:
        return asdict(self)


@dataclass
class SensorConfig:
    sensor_type: str = ""  # "realsense", "kinect", "generic"
    serial: str = ""
    firmware: str = ""
    imu_rate_hz: float = 0.0
    rgb_rate_hz: float = 0.0
    depth_rate_hz: float = 0.0

    def to_yaml(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


def parse_camera_info_message(data: bytes) -> CameraIntrinsics | None:
    """Parse CameraInfo CDR bytes into intrinsics.

    RealSense CameraInfo layout (verified against actual data):
    - seq(4), stamp.sec(4), stamp.nsec(4) = 12 bytes
    - frame_id: len(4) + 27 chars + null = 32 bytes, then height at 44
    - height(4), width(4)
    - distortion_model: len(4) + 20 chars (18 + 2 null padding)
    - binning_x(4), binning_y(4)
    - D[8] (no length prefix - directly 8 float64 values)
    - K[9], R[9], P[12]
    """
    try:
        offset = 0
        # seq (4 bytes)
        offset += 4
        # stamp: sec(4) + nsec(4) = 8 bytes
        offset += 8

        # frame_id string: len(4) + content (27 chars + null)
        str_len = int.from_bytes(data[offset:offset+4], 'little')
        offset += 4 + str_len
        # Height is at byte 44 (not 8-byte aligned after frame_id)
        offset = 44

        # height, width (4 bytes each)
        height = int.from_bytes(data[offset:offset+4], 'little')
        offset += 4
        width = int.from_bytes(data[offset:offset+4], 'little')
        offset += 4

        # distortion_model string: len(4) + content (18 chars + 2 null = 20 bytes)
        dm_len = int.from_bytes(data[offset:offset+4], 'little')
        offset += 4 + dm_len

        # binning_x, binning_y (4 bytes each)
        binning_x = int.from_bytes(data[offset:offset+4], 'little')
        offset += 4
        binning_y = int.from_bytes(data[offset:offset+4], 'little')
        offset += 4

        # D array: 8 float64 values directly (no length prefix)
        # For rational_polynomial model, D has 5-8 coefficients
        n_d = 8  # RealSense uses 8 coefficients
        D = np.frombuffer(data[offset:offset+n_d*8], dtype=np.float64).copy()
        offset += n_d * 8

        # K[9] - 9 float64 = 72 bytes
        K = np.frombuffer(data[offset:offset+72], dtype=np.float64).copy()
        offset += 72

        # R[9] - 9 float64 = 72 bytes
        R = np.frombuffer(data[offset:offset+72], dtype=np.float64).copy()
        offset += 72

        # P[12] - 12 float64 = 96 bytes
        P = np.frombuffer(data[offset:offset+96], dtype=np.float64).copy()

        return CameraIntrinsics(
            model="pinhole" if len(D) == 0 else "opencv",
            width=width,
            height=height,
            fx=K[0] if len(K) > 0 else P[0],
            fy=K[4] if len(K) > 4 else P[5],
            cx=K[2] if len(K) > 2 else P[2],
            cy=K[5] if len(K) > 5 else P[6],
            k1=D[0] if len(D) > 0 else 0.0,
            k2=D[1] if len(D) > 1 else 0.0,
            p1=D[2] if len(D) > 2 else 0.0,
            p2=D[3] if len(D) > 3 else 0.0,
            k3=D[4] if len(D) > 4 else 0.0,
        )
    except Exception as e:
        return None


def parse_realsense_metadata(data: bytes) -> dict:
    """Parse RealSense metadata messages."""
    # RealSense Metadata format: key-value pairs
    result = {}
    try:
        offset = 0
        # Each entry: 4 byte key length, key string, 8 byte value type, value
        while offset < len(data) - 4:
            key_len = int.from_bytes(data[offset:offset+4], 'little')
            offset += 4
            if key_len == 0 or offset + key_len > len(data):
                break
            key = data[offset:offset+key_len].decode('utf-8', errors='ignore')
            offset += key_len
            if offset + 8 > len(data):
                break
            value_type = int.from_bytes(data[offset:offset+8], 'little')
            offset += 8
            # type 1 = int, 2 = float, 3 = string
            if value_type == 1:
                val = int.from_bytes(data[offset:offset+8], 'little')
                offset += 8
            elif value_type == 2:
                val = float.from_buffer(data[offset:offset+8])
                offset += 8
            elif value_type == 3:
                str_len = int.from_bytes(data[offset:offset+4], 'little')
                offset += 4
                val = data[offset:offset+str_len].decode('utf-8', errors='ignore')
                offset += str_len
            else:
                break
            result[key] = val
    except Exception:
        pass
    return result


def detect_depth_format(topics: dict[str, str], topic_name: str, metadata: dict) -> DepthFormat:
    """Detect depth format from topic name and metadata.
    
    RealSense depth is typically in millimeters with scale 1.0.
    """
    topic_lower = topic_name.lower()
    
    if "realsense" in metadata.get("sensor_type", "").lower():
        return DepthFormat(
            unit="millimeters",
            scale=1.0,
            description="RealSense depth camera, millimeters, no scaling"
        )
    
    if "aligned" in topic_lower or "register" in topic_lower:
        # Aligned depth is typically metric (meters or millimeters)
        return DepthFormat(
            unit="millimeters",
            scale=1.0,
            description="Aligned depth to RGB, millimeters"
        )
    
    if "depth" in topic_lower:
        # Default RealSense
        return DepthFormat(
            unit="millimeters",
            scale=1.0,
            description="Depth image, millimeters"
        )
    
    return DepthFormat(unit="unknown", scale=1.0)


def build_extrinsics_from_tf(tf_message: dict) -> CameraExtrinsics:
    """Build camera extrinsics from TF message."""
    # TF format: header (frame_id), child_frame_id, transform (translation xyz, rotation quaternion)
    return CameraExtrinsics(
        translation=tf_message.get("translation", (0.0, 0.0, 0.0)),
        rotation_quaternion=tf_message.get("rotation", (1.0, 0.0, 0.0, 0.0)),
    )


def save_intrinsics(intrinsics: CameraIntrinsics, output_path: Path):
    with open(output_path, 'w') as f:
        yaml.dump(intrinsics.to_yaml(), f, default_flow_style=False)


def save_extrinsics(extrinsics: CameraExtrinsics, output_path: Path):
    with open(output_path, 'w') as f:
        yaml.dump(extrinsics.to_yaml(), f, default_flow_style=False)


def save_depth_format(depth_format: DepthFormat, output_path: Path):
    with open(output_path, 'w') as f:
        yaml.dump(depth_format.to_yaml(), f, default_flow_style=False)


def save_sensor_config(sensor: SensorConfig, output_path: Path):
    with open(output_path, 'w') as f:
        yaml.dump(sensor.to_yaml(), f, default_flow_style=False)