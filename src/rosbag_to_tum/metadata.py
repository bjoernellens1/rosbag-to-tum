"""Comprehensive metadata extraction from bag."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from .camera import CameraIntrinsics, CameraExtrinsics, DepthFormat, SensorConfig, parse_camera_info_message, detect_depth_format


@dataclass
class BagMetadata:
    bag_name: str = ""
    bag_path: str = ""
    duration_ns: int = 0
    start_time_ns: int = 0
    end_time_ns: int = 0
    message_count: int = 0
    topics: dict[str, str] = field(default_factory=dict)  # topic -> type
    rgb_topic: str = ""
    depth_topic: str = ""
    aligned_depth_topic: str = ""
    camera_info_topic: str = ""
    depth_camera_info_topic: str = ""
    imu_topic: str = ""
    odom_topic: str = ""
    tf_topic: str = ""
    has_odom: bool = False
    has_tf: bool = False
    has_imu: bool = False
    sensor_type: str = "unknown"
    intrinsics_rgb: Optional[CameraIntrinsics] = None
    intrinsics_depth: Optional[CameraIntrinsics] = None
    extrinsics: Optional[CameraExtrinsics] = None
    depth_format: Optional[DepthFormat] = None
    sensor_config: Optional[SensorConfig] = None

    def to_yaml(self) -> dict:
        result = {
            "bag_name": self.bag_name,
            "bag_path": self.bag_path,
            "duration_s": self.duration_ns / 1e9 if self.duration_ns else 0,
            "start_time": self.start_time_ns,
            "end_time": self.end_time_ns,
            "message_count": self.message_count,
            "topics": self.topics,
            "topic_assignments": {
                "rgb": self.rgb_topic,
                "depth": self.depth_topic,
                "aligned_depth": self.aligned_depth_topic,
                "camera_info": self.camera_info_topic,
                "depth_camera_info": self.depth_camera_info_topic,
                "imu": self.imu_topic,
                "odom": self.odom_topic,
                "tf": self.tf_topic,
            },
            "sensor_type": self.sensor_type,
            "capabilities": {
                "has_odom": self.has_odom,
                "has_tf": self.has_tf,
                "has_imu": self.has_imu,
            },
        }
        
        if self.intrinsics_rgb:
            result["intrinsics_rgb"] = self.intrinsics_rgb.to_yaml()
        if self.intrinsics_depth:
            result["intrinsics_depth"] = self.intrinsics_depth.to_yaml()
        if self.extrinsics:
            result["extrinsics"] = self.extrinsics.to_yaml()
        if self.depth_format:
            result["depth_format"] = self.depth_format.to_yaml()
        if self.sensor_config:
            result["sensor_config"] = self.sensor_config.to_yaml()
        
        return result

    def save(self, path: Path):
        with open(path, 'w') as f:
            yaml.dump(self.to_yaml(), f, default_flow_style=False, sort_keys=False)


def detect_sensor_type(topics: dict[str, str]) -> str:
    """Detect sensor type from topic names."""
    topics_str = " ".join(topics.keys()).lower()

    if "orbbec" in topics_str or "femto" in topics_str:
        return "orbbec_femto_bolt"
    if "realsense" in topics_str or "rs" in topics_str:
        return "realsense"
    if "kinect" in topics_str:
        return "kinect"
    if "azure" in topics_str:
        return "azure_kinect"
    return "unknown"


def analyze_bag(reader, topics: dict[str, str]) -> BagMetadata:
    """Analyze bag and extract all metadata."""
    from .reader import detect_image_topics, detect_camera_info_topics
    
    metadata = BagMetadata()
    metadata.topics = topics
    
    # Detect image topics
    image_roles = detect_image_topics(topics)
    for topic, role in image_roles.items():
        if role == "rgb":
            metadata.rgb_topic = topic
        elif role == "depth":
            metadata.depth_topic = topic
        elif role == "aligned_depth":
            metadata.aligned_depth_topic = topic
    
    # Detect camera info topics
    cam_info_roles = detect_camera_info_topics(topics)
    for topic, role in cam_info_roles.items():
        if role == "rgb_cam_info":
            metadata.camera_info_topic = topic
        elif role == "depth_cam_info":
            metadata.depth_camera_info_topic = topic
    
    # Check for odom, tf, imu
    for topic in topics:
        t = topic.lower()
        if "odom" in t:
            metadata.odom_topic = topic
            metadata.has_odom = True
        if "tf" in t and "static" not in t:
            metadata.tf_topic = topic
            metadata.has_tf = True
        if "imu" in t:
            metadata.imu_topic = topic
            metadata.has_imu = True
    
    # Detect sensor type
    metadata.sensor_type = detect_sensor_type(topics)
    
    return metadata


def extract_camera_params(reader, metadata: BagMetadata) -> BagMetadata:
    """Extract camera intrinsics, extrinsics from bag messages."""
    
    # Collect CameraInfo messages
    rgb_cam_info_data = None
    depth_cam_info_data = None
    tf_messages = []
    realsense_metadata: dict = {}
    
    for topic, ts, data in reader.iter_messages():
        if metadata.camera_info_topic and topic == metadata.camera_info_topic:
            rgb_cam_info_data = parse_camera_info_message(data)
        elif metadata.depth_camera_info_topic and topic == metadata.depth_camera_info_topic:
            depth_cam_info_data = parse_camera_info_message(data)
        elif "metadata" in topic.lower() and "realsense" in metadata.sensor_type:
            # Parse realsense metadata
            pass
        elif topic == metadata.tf_topic:
            from .trajectory import parse_tf_message
            tf_messages.extend(parse_tf_message(data))
    
    if rgb_cam_info_data:
        metadata.intrinsics_rgb = rgb_cam_info_data
    
    if depth_cam_info_data:
        metadata.intrinsics_depth = depth_cam_info_data
    
    # Default depth format
    metadata.depth_format = DepthFormat(
        unit="millimeters",
        scale=1.0,
        description="Depth in millimeters, scale=1.0 (no TUM 5000 factor)"
    )
    
    # Try to build extrinsics from TF
    if tf_messages:
        for parent, child, ts, xyz, quat in tf_messages:
            if "camera" in child and "color" in child.lower():
                metadata.extrinsics = CameraExtrinsics(
                    translation=xyz,
                    rotation_quaternion=quat,
                )
                break
    
    return metadata


def create_sensor_config(metadata: BagMetadata, reader) -> SensorConfig:
    """Create sensor configuration from bag metadata."""
    config = SensorConfig()
    config.sensor_type = metadata.sensor_type
    
    # Estimate rates from message counts and duration
    if metadata.duration_ns > 0:
        dur_s = metadata.duration_ns / 1e9
        for topic, msg_count in metadata.topics.items():
            if "color" in topic.lower() and "image" in topic.lower():
                config.rgb_rate_hz = msg_count / dur_s
            elif "depth" in topic.lower() and "image" in topic.lower():
                config.depth_rate_hz = msg_count / dur_s
            elif "imu" in topic.lower():
                config.imu_rate_hz = msg_count / dur_s
    
    return config