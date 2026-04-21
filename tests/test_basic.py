"""Test basic reader functionality."""

import pytest
from pathlib import Path


def test_import():
    from rosbag_to_tum import __version__
    assert __version__ == "0.1.0"


def test_camera_info_parse():
    from rosbag_to_tum.camera import parse_camera_info_message
    # Just verify the function exists and handles None
    result = parse_camera_info_message(b"")
    assert result is None


def test_depth_format():
    from rosbag_to_tum.camera import DepthFormat
    df = DepthFormat(unit="millimeters", scale=1.0)
    assert df.unit == "millimeters"
    assert df.scale == 1.0
    yaml = df.to_yaml()
    assert yaml["unit"] == "millimeters"


def test_visual_tracker_config():
    from rosbag_to_tum.visual_tracker import VisualTrackConfig
    config = VisualTrackConfig(max_features=100)
    assert config.max_features == 100


def test_tum_formatter_init(tmp_path):
    from rosbag_to_tum.formatter import TUMFormatter
    from rosbag_to_tum.camera import CameraIntrinsics
    
    intrinsics = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240, width=640, height=480)
    formatter = TUMFormatter(tmp_path, "test_bag", intrinsics=intrinsics)
    
    assert formatter.rgb_dir.exists()
    assert formatter.depth_dir.exists()