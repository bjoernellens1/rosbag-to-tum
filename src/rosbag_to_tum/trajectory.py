"""Extract camera trajectory from TF, odometry, or visual tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


@dataclass
class TrajectoryEntry:
    timestamp_ns: int
    tx: float
    ty: float
    tz: float
    qw: float
    qx: float
    qy: float
    qz: float
    source: str = "unknown"  # "tf", "odom", "visual"


@dataclass
class Trajectory:
    entries: list[TrajectoryEntry]
    source: str  # "tf", "odom", "visual", "mixed"

    def sort_by_time(self):
        self.entries.sort(key=lambda e: e.timestamp_ns)

    def save(self, path: Path, name: str = "groundtruth"):
        """Save in TUM format: timestamp tx ty tz qx qy qz qw"""
        lines = []
        for e in self.entries:
            ts_sec = e.timestamp_ns / 1e9
            lines.append(f"{ts_sec:.6f} {e.tx:.6f} {e.ty:.6f} {e.tz:.6f} {e.qx:.6f} {e.qy:.6f} {e.qz:.6f} {e.qw:.6f}")

        with open(path, 'w') as f:
            f.write(f"# {name} trajectory - source: {self.source}\n")
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            f.write("\n".join(lines))

    def save_multiple(self, path: Path, groundtruth_name: str, trajectory_name: str):
        """Save as groundtruth.txt and trajectory.txt based on source."""
        gt_lines = ["# timestamp tx ty tz qx qy qz qw"]
        traj_lines = ["# timestamp tx ty tz qx qy qz qw"]

        for e in self.entries:
            ts_sec = e.timestamp_ns / 1e9
            line = f"{ts_sec:.6f} {e.tx:.6f} {e.ty:.6f} {e.tz:.6f} {e.qx:.6f} {e.qy:.6f} {e.qz:.6f} {e.qw:.6f}"
            if e.source == "visual":
                traj_lines.append(line)
            else:
                gt_lines.append(line)

        gt_path = path.parent / "groundtruth.txt"
        traj_path = path.parent / "trajectory.txt"

        with open(gt_path, 'w') as f:
            f.write(f"# Groundtruth trajectory - source: odometry/TF\n")
            f.write("\n".join(gt_lines))

        with open(traj_path, 'w') as f:
            f.write(f"# Estimated trajectory - source: visual tracker\n")
            f.write("\n".join(traj_lines))


def parse_tf_message(data: bytes) -> list[tuple[str, str, int, tuple, tuple]]:
    """Parse TFMessage CDR bytes.

    Returns: [(parent_frame, child_frame, timestamp_ns, translation_xyz, quaternion_xyzw)]
    """
    transforms = []
    try:
        offset = 0

        if len(data) < 8:
            return []

        # Skip seq (4 bytes)
        offset += 4

        # Array count (4 bytes)
        count = int.from_bytes(data[offset:offset+4], 'little')
        offset += 4

        for _ in range(count):
            if offset >= len(data):
                break

            # stamp sec + nsec (8 bytes total)
            sec = int.from_bytes(data[offset:offset+4], 'little')
            nsec = int.from_bytes(data[offset+4:offset+8], 'little')
            timestamp_ns = sec * 1_000_000_000 + nsec
            offset += 8

            # frame_id string (parent frame)
            frame_len = int.from_bytes(data[offset:offset+4], 'little')
            offset += 4
            parent_frame = data[offset:offset+frame_len].decode('utf-8', errors='ignore').rstrip('\x00')
            offset += frame_len
            # Align to 4-byte boundary
            offset += (4 - (frame_len % 4)) % 4

            # child_frame_id string
            child_len = int.from_bytes(data[offset:offset+4], 'little')
            offset += 4
            child_frame = data[offset:offset+child_len].decode('utf-8', errors='ignore').rstrip('\x00')
            offset += child_len
            # Align to 4-byte boundary
            offset += (4 - (child_len % 4)) % 4

            # transform translation xyz (3 * 8 bytes)
            tx = struct.unpack('<d', data[offset:offset+8])[0]
            ty = struct.unpack('<d', data[offset+8:offset+16])[0]
            tz = struct.unpack('<d', data[offset+16:offset+24])[0]
            offset += 24

            # rotation quaternion (4 * 8 bytes) - ROS uses xyzw order in buffer
            qx = struct.unpack('<d', data[offset:offset+8])[0]
            qy = struct.unpack('<d', data[offset+8:offset+16])[0]
            qz = struct.unpack('<d', data[offset+16:offset+24])[0]
            qw = struct.unpack('<d', data[offset+24:offset+32])[0]
            offset += 32

            transforms.append((parent_frame, child_frame, timestamp_ns, (tx, ty, tz), (qw, qx, qy, qz)))

    except Exception:
        pass

    return transforms


def parse_odometry_message(data: bytes) -> tuple[int, tuple, tuple] | None:
    """Parse Odometry CDR bytes.

    RealSense odometry layout:
    - seq(4), stamp.sec(4), stamp.nsec(4) = 12 bytes
    - frame_id: len(4) + content + null padding
    - child_frame_id: len(4) + content + null padding
    - pose: position(24 bytes) + orientation(32 bytes)

    Returns: (timestamp_ns, pose_xyz, pose_quaternion)
    """
    try:
        offset = 0
        offset += 4  # seq
        sec = int.from_bytes(data[offset:offset+4], 'little', signed=True)
        nsec = int.from_bytes(data[offset+4:offset+8], 'little')
        timestamp_ns = sec * 1_000_000_000 + nsec
        offset += 8

        # frame_id string
        frame_len = int.from_bytes(data[offset:offset+4], 'little')
        offset += 4
        frame_id = data[offset:offset+frame_len].decode('utf-8', errors='ignore').rstrip('\x00')
        offset += frame_len
        # Align to 4-byte boundary
        offset += (4 - (frame_len % 4)) % 4

        # child_frame_id string
        child_len = int.from_bytes(data[offset:offset+4], 'little')
        offset += 4
        child_frame = data[offset:offset+child_len].decode('utf-8', errors='ignore').rstrip('\x00')
        offset += child_len
        # Align to 4-byte boundary
        offset += (4 - (child_len % 4)) % 4

        # pose position (24 bytes = 3 * 8)
        tx = struct.unpack('<d', data[offset:offset+8])[0]
        ty = struct.unpack('<d', data[offset+8:offset+16])[0]
        tz = struct.unpack('<d', data[offset+16:offset+24])[0]
        offset += 24

        # pose orientation quaternion (32 bytes = 4 * 8)
        qx = struct.unpack('<d', data[offset:offset+8])[0]
        qy = struct.unpack('<d', data[offset+8:offset+16])[0]
        qz = struct.unpack('<d', data[offset+16:offset+24])[0]
        qw = struct.unpack('<d', data[offset+24:offset+32])[0]

        return timestamp_ns, (tx, ty, tz), (qw, qx, qy, qz)

    except Exception:
        return None


def extract_trajectory_from_tf_odom(reader, topics: list[str]) -> Trajectory | None:
    """Extract trajectory from TF and/or odometry topics."""
    tf_topic = None
    odom_topic = None

    for t in topics:
        if "tf" in t.lower() and "static" not in t.lower():
            tf_topic = t
        if "odom" in t.lower():
            odom_topic = t

    if not tf_topic and not odom_topic:
        return None

    # Collect poses from odometry first
    odom_entries: list[TrajectoryEntry] = []
    tf_entries: list[TrajectoryEntry] = []

    for topic, ts, data in reader.iter_messages():
        if odom_topic and topic == odom_topic:
            result = parse_odometry_message(data)
            if result:
                timestamp_ns, (tx, ty, tz), (qw, qx, qy, qz) = result
                odom_entries.append(TrajectoryEntry(
                    timestamp_ns=timestamp_ns,
                    tx=tx, ty=ty, tz=tz,
                    qw=qw, qx=qx, qy=qy, qz=qz,
                    source="odom"
                ))

        if tf_topic and topic == tf_topic:
            transforms = parse_tf_message(data)
            for parent, child, ts_ns, xyz, quat in transforms:
                # Look for camera-to-world or base-to-world transforms
                if "camera" in child or "base" in child or "odom" in child:
                    tx, ty, tz = xyz
                    qw, qx, qy, qz = quat
                    tf_entries.append(TrajectoryEntry(
                        timestamp_ns=ts_ns,
                        tx=tx, ty=ty, tz=tz,
                        qw=qw, qx=qx, qy=qy, qz=qz,
                        source="tf"
                    ))

    # Combine trajectories
    if odom_entries:
        traj = Trajectory(entries=odom_entries, source="odom")
        traj.sort_by_time()
        return traj

    if tf_entries:
        traj = Trajectory(entries=tf_entries, source="tf")
        traj.sort_by_time()
        return traj

    return None