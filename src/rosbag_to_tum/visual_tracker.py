"""Visual tracking fallback when no odometry/TF is available.

Uses OpenCV optical flow + PnP solver for visual odometry with known intrinsics.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from .trajectory import TrajectoryEntry, Trajectory


@dataclass
class VisualTrackConfig:
    max_features: int = 500
    min_track_length: int = 15
    pnp_ransac_threshold: float = 3.0
    max_pose_delta_m: float = 0.5  # sanity check: max translation between frames
    depth_scale: float = 1000.0  # depth in mm, output in meters


def rotation_vector_to_quaternion(rvec: np.ndarray) -> tuple[float, float, float, float]:
    """Convert OpenCV rotation vector to quaternion (qw, qx, qy, qz)."""
    rot_mat, _ = cv2.Rodrigues(rvec)
    # Extract quaternion from rotation matrix
    trace = np.trace(rot_mat)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (rot_mat[2, 1] - rot_mat[1, 2]) * s
        qy = (rot_mat[0, 2] - rot_mat[2, 0]) * s
        qz = (rot_mat[1, 0] - rot_mat[0, 1]) * s
    else:
        if rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
            qw = (rot_mat[2, 1] - rot_mat[1, 2]) / s
            qx = 0.25 * s
            qy = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            qz = (rot_mat[0, 2] + rot_mat[2, 0]) / s
        elif rot_mat[1, 1] > rot_mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
            qw = (rot_mat[0, 2] - rot_mat[2, 0]) / s
            qx = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            qy = 0.25 * s
            qz = (rot_mat[1, 2] + rot_mat[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
            qw = (rot_mat[1, 0] - rot_mat[0, 1]) / s
            qx = (rot_mat[0, 2] + rot_mat[2, 0]) / s
            qy = (rot_mat[1, 2] + rot_mat[2, 1]) / s
            qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


class VisualTracker:
    """Visual odometry tracker using optical flow + PnP with known intrinsics."""

    def __init__(self, config: VisualTrackConfig = None, intrinsics: list = None):
        self.config = config or VisualTrackConfig()
        self.intrinsics = intrinsics  # [fx, fy, cx, cy]
        self.keypoints_prev: list = []
        self.pts_3d_prev: list = []  # 3D points from previous frame
        self.frame_prev_gray = None
        self.trajectory_entries: list[TrajectoryEntry] = []
        self.frame_idx = 0
        self.depth_scale = self.config.depth_scale
        # Absolute pose tracking (chain PnP results)
        self.absolute_rvec = np.zeros(3)  # accumulated rotation
        self.absolute_tvec = np.zeros(3)  # accumulated translation

    def _backproject_to_3d(self, pts_2d: np.ndarray, depth_img: np.ndarray) -> tuple[list, list]:
        """Backproject 2D image points to 3D using depth and intrinsics.

        Returns (pts_3d, valid_indices) where valid_indices are indices into pts_2d
        that had valid depth.
        """
        fx, fy, cx, cy = self.intrinsics
        pts_3d = []
        valid_indices = []
        for idx, pt in enumerate(pts_2d):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < depth_img.shape[0] and 0 <= x < depth_img.shape[1]:
                d = depth_img[y, x]
                if d > 0:
                    z = d / self.depth_scale  # mm to meters
                    x_3d = (x - cx) * z / fx
                    y_3d = (y - cy) * z / fy
                    pts_3d.append([x_3d, y_3d, z])
                    valid_indices.append(idx)
        return pts_3d, valid_indices

    def _detect_keypoints(self, gray: np.ndarray) -> list:
        """Detect ORB features and return their coordinates."""
        orb = cv2.ORB_create(nfeatures=self.config.max_features)
        keypoints = orb.detect(gray, None)
        # Return just the coordinates as a list of (x, y) tuples
        return [(kp.pt[0], kp.pt[1]) for kp in keypoints]

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
            pts_prev = np.array(self.keypoints_prev, dtype=np.float32).reshape(-1, 1, 2)

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

            if len(pts_valid) > self.config.min_track_length and depth_img is not None:
                # Backproject previous points to 3D (filtering out invalid depth)
                pts_3d, valid_indices = self._backproject_to_3d(pts_prev_valid.reshape(-1, 2), depth_img)

                if len(pts_3d) >= 6:
                    pts_3d_arr = np.array(pts_3d, dtype=np.float32)
                    pts_valid_2d = pts_valid.reshape(-1, 2)[valid_indices]
                    # Use PnP to estimate relative pose change
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        pts_3d_arr,
                        pts_valid_2d,
                        np.array([[self.intrinsics[0], 0, self.intrinsics[2]],
                                  [0, self.intrinsics[1], self.intrinsics[3]],
                                  [0, 0, 1]], dtype=np.float32),
                        distCoeffs=np.zeros(5, dtype=np.float32),
                        iterationsCount=100,
                        reprojectionError=self.config.pnp_ransac_threshold,
                        confidence=0.99
                    )

                    if success:
                        # Update absolute pose by chaining
                        delta_t = tvec.flatten()
                        delta_r = rvec.flatten()

                        # Sanity check on translation magnitude
                        t_mag = np.linalg.norm(delta_t)
                        if t_mag < self.config.max_pose_delta_m:
                            # Accumulate pose
                            self.absolute_tvec += self._rotate_vector(delta_t, self.absolute_rvec)
                            self.absolute_rvec += delta_r

                            # Convert to quaternion
                            qw, qx, qy, qz = rotation_vector_to_quaternion(self.absolute_rvec)

                            self.trajectory_entries.append(TrajectoryEntry(
                                timestamp_ns=timestamp_ns,
                                tx=float(self.absolute_tvec[0]),
                                ty=float(self.absolute_tvec[1]),
                                tz=float(self.absolute_tvec[2]),
                                qw=qw, qx=qx, qy=qy, qz=qz,
                                source="visual"
                            ))

        # Update for next iteration
        self.keypoints_prev = self._detect_keypoints(gray)
        self.frame_prev_gray = gray.copy()
        self.frame_idx += 1

    def _rotate_vector(self, v: np.ndarray, rvec: np.ndarray) -> np.ndarray:
        """Rotate vector v by rotation vector rvec."""
        rot_mat, _ = cv2.Rodrigues(rvec)
        return rot_mat @ v

    def get_trajectory(self) -> Trajectory:
        """Return estimated trajectory."""
        traj = Trajectory(entries=self.trajectory_entries, source="visual")
        traj.sort_by_time()
        return traj


@dataclass
class VisualTrackerWithFiles:
    """Visual tracker that operates on extracted image files."""

    config: VisualTrackConfig = field(default_factory=VisualTrackConfig)
    intrinsics: list = field(default=None)

    def run(self, rgb_dir: Path, depth_dir: Path | None, max_frames: int = 1000) -> Trajectory:
        """Estimate trajectory from extracted PNG files."""
        tracker = VisualTracker(config=self.config, intrinsics=self.intrinsics)

        # Pre-load depth timestamps for efficient matching
        depth_files = sorted(depth_dir.glob("*.png")) if depth_dir else []
        depth_timestamps = []
        if depth_files:
            for dp in depth_files:
                try:
                    ts_sec = float(dp.stem.split("_")[0])
                    depth_timestamps.append((ts_sec, dp))
                except (ValueError, IndexError):
                    pass

        rgb_files = sorted(rgb_dir.glob("*.png"))[:max_frames]

        for i, rgb_path in enumerate(rgb_files):
            color_img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if color_img is None:
                continue

            # Extract timestamp from filename
            try:
                stem = rgb_path.stem.split("_")[0]
                ts_sec = float(stem)
                ts_ns = int(ts_sec * 1e9)
            except (ValueError, IndexError):
                ts_ns = i * 33333333

            # Find closest depth frame by timestamp
            depth_img = None
            if depth_timestamps:
                target_ts = ts_sec
                closest_dp = min(depth_timestamps, key=lambda x: abs(x[0] - target_ts))
                # Only use if within 0.5 seconds
                if abs(closest_dp[0] - target_ts) < 0.5:
                    depth_img = cv2.imread(str(closest_dp[1]), cv2.IMREAD_ANYDEPTH)

            tracker.process_frame(color_img, depth_img, ts_ns)

        return tracker.get_trajectory()


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

    config = VisualTrackConfig()
    tracker = VisualTracker(config=config, intrinsics=intrinsics)

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

                # Get corresponding depth if available - need to buffer
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


def estimate_trajectory_from_files(
    rgb_dir: Path,
    depth_dir: Path | None,
    intrinsics: list,
    max_frames: int = 1000
) -> Trajectory:
    """Estimate trajectory from extracted PNG files.

    Args:
        rgb_dir: Path to directory containing RGB PNG files
        depth_dir: Optional path to depth PNG files
        intrinsics: Camera intrinsics [fx, fy, cx, cy]
        max_frames: Maximum frames to process

    Returns:
        Trajectory with estimated poses
    """
    tracker_impl = VisualTrackerWithFiles(intrinsics=intrinsics)
    return tracker_impl.run(rgb_dir, depth_dir, max_frames)