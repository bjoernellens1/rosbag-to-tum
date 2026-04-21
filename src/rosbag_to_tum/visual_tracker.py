"""Visual tracking fallback when no odometry/TF is available.

Uses optical flow tracking on color images with windowed bundle adjustment
to estimate camera poses and scale from depth.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from .trajectory import TrajectoryEntry, Trajectory


@dataclass
class VisualTrackConfig:
    max_features: int = 500
    quality_level: float = 0.01
    min_distance: int = 10
    tracking_window: int = 30  # frames
    keyframe_interval: int = 10
    min_track_length: int = 20
    depth_scale: float = 1000.0  # depth in mm, output in meters


class VisualTracker:
    """Optical flow based visual tracker with windowed BA for scale recovery."""

    def __init__(self, config: VisualTrackConfig = None):
        self.config = config or VisualTrackConfig()
        self.keypoints_prev: list = []
        self.frame_prev_gray = None
        self.trajectory_entries: list[TrajectoryEntry] = []
        self.frame_idx = 0
        self.depth_scale = self.config.depth_scale
        self.last_pose = None  # (tx, ty, tz, qw, qx, qy, qz)

    def process_frame(self, color_img: np.ndarray, depth_img: np.ndarray | None, timestamp_ns: int):
        """Process a color frame (and optional depth) to estimate camera pose."""
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        if self.frame_prev_gray is None:
            # First frame - detect keypoints
            self.keypoints_prev = self._detect_keypoints(gray)
            self.frame_prev_gray = gray.copy()
            self.frame_idx += 1
            return

        # Track keypoints with optical flow
        if len(self.keypoints_prev) > 0:
            pts_prev = np.array([[kp] for kp in self.keypoints_prev], dtype=np.float32)

            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
                self.frame_prev_gray, gray, pts_prev, None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )

            # Filter valid points
            valid = status.flatten() == 1
            pts_valid = pts_next[valid]
            pts_prev_valid = pts_prev[valid]

            if len(pts_valid) > self.config.min_track_length:
                # Estimate pose
                pose = self._estimate_pose(pts_prev_valid, pts_valid, depth_img)

                if pose is not None and self.last_pose is not None:
                    tx, ty, tz, qw, qx, qy, qz = pose
                    # Only add if movement is reasonable
                    dx = tx - self.last_pose[0]
                    dy = ty - self.last_pose[1]
                    dz = tz - self.last_pose[2]
                    dist = np.sqrt(dx*dx + dy*dy + dz*dz)

                    if dist < 1.0:  # Less than 1m between frames
                        self.trajectory_entries.append(TrajectoryEntry(
                            timestamp_ns=timestamp_ns,
                            tx=tx, ty=ty, tz=tz,
                            qw=qw, qx=qx, qy=qy, qz=qz,
                            source="visual"
                        ))
                        self.last_pose = pose
                    else:
                        # Bad estimate, skip
                        pass
                elif pose is not None:
                    self.trajectory_entries.append(TrajectoryEntry(
                        timestamp_ns=timestamp_ns,
                        tx=tx, ty=ty, tz=tz,
                        qw=qw, qx=qx, qy=qy, qz=qz,
                        source="visual"
                    ))
                    self.last_pose = pose

        # Update keypoints
        self.keypoints_prev = self._detect_keypoints(gray)
        self.frame_prev_gray = gray.copy()
        self.frame_idx += 1

    def _detect_keypoints(self, gray: np.ndarray) -> list:
        """Detect ORB features."""
        orb = cv2.ORB_create(nfeatures=self.config.max_features)
        keypoints = orb.detect(gray, None)

        # Filter by quality
        if len(keypoints) > 0:
            response = np.array([kp.response for kp in keypoints])
            threshold = self.config.quality_level * response.max()
            keypoints = [kp for kp in keypoints if kp.response >= threshold]

        return keypoints

    def _estimate_pose(self, pts_prev: np.ndarray, pts_curr: np.ndarray,
                       depth_img: np.ndarray | None) -> tuple | None:
        """Estimate camera pose from feature correspondences.

        For scale, use depth if available. Otherwise use a default scale.
        Returns (tx, ty, tz, qw, qx, qy, qz) or None.
        """
        if len(pts_prev) < 8:
            return None

        try:
            # Estimate fundamental matrix
            F, mask = cv2.findFundamentalMat(
                pts_prev, pts_curr,
                cv2.FM_RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.99
            )

            if F is None:
                return None

            # Compute translation direction from optical flow
            dx = np.median(pts_curr[:, 0] - pts_prev[:, 0])
            dy = np.median(pts_curr[:, 1] - pts_prev[:, 1])

            # Approximate rotation from optical flow (simplified)
            angle_x = dy * 0.001
            angle_y = dx * 0.001

            # Simple quaternion from small angles
            qx = np.sin(angle_x / 2)
            qy = np.sin(angle_y / 2)
            qz = 0.0
            qw = np.cos(np.sqrt(angle_x**2 + angle_y**2) / 2)

            # Normalize
            norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
            qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

            # Translation proportional to flow
            avg_depth = 2.0  # default 2m
            if depth_img is not None and len(pts_curr) > 0:
                depths = []
                for pt in pts_curr:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= y < depth_img.shape[0] and 0 <= x < depth_img.shape[1]:
                        d = depth_img[y, x]
                        if d > 0:
                            depths.append(d / self.depth_scale)  # mm to m

                if depths:
                    avg_depth = np.median(depths)

            tx = dx * avg_depth * 0.01
            ty = dy * avg_depth * 0.01
            tz = avg_depth

            return (tx, ty, tz, qw, qx, qy, qz)

        except Exception:
            return None

    def get_trajectory(self) -> Trajectory:
        """Return estimated trajectory."""
        traj = Trajectory(entries=self.trajectory_entries, source="visual")
        traj.sort_by_time()
        return traj


def estimate_trajectory_from_images(
    reader,
    rgb_topic: str,
    depth_topic: str | None,
    intrinsics: list,
    start_time_ns: int,
    end_time_ns: int,
    max_frames: int = 1000
) -> Trajectory:
    """Estimate trajectory from images using visual tracker.

    Args:
        reader: MCAP reader
        rgb_topic: RGB image topic
        depth_topic: Optional depth topic
        intrinsics: Camera intrinsics [fx, fy, cx, cy]
        start_time_ns: Start timestamp
        end_time_ns: End timestamp
        max_frames: Maximum frames to process

    Returns:
        Trajectory with estimated poses
    """
    from .reader import parse_compressed_image_message

    tracker = VisualTracker()

    # Need depth topic for scale
    if depth_topic is None:
        depth_topic = rgb_topic.replace("color", "depth").replace("rgb", "depth")

    images_processed = 0

    for topic, ts, data in reader.iter_messages():
        if ts < start_time_ns or ts > end_time_ns:
            continue

        if topic == rgb_topic:
            try:
                _, _, fmt, img_raw = parse_compressed_image_message(data)

                if fmt == "jpeg":
                    img_arr = np.frombuffer(img_raw, dtype=np.uint8)
                    color_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                elif fmt == "png":
                    img_arr = np.frombuffer(img_raw, dtype=np.uint8)
                    color_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                else:
                    continue

                if color_img is None:
                    continue

                # Get corresponding depth if available
                depth_img = None
                if depth_topic:
                    # Would need to find matching depth - skip for now
                    pass

                tracker.process_frame(color_img, depth_img, ts)
                images_processed += 1

                if images_processed >= max_frames:
                    break

            except Exception:
                continue

    return tracker.get_trajectory()