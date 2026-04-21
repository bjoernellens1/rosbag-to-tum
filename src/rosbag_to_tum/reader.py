"""Read ROS bags (MCAP format) and extract messages - no rosidl dependency."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np


@dataclass
class BagTopic:
    name: str
    type: str
    message_count: int


@dataclass
class CompressedImage:
    timestamp_ns: int
    topic: str
    data: bytes  # JPEG/compressed bytes
    format: str  # "jpeg" or "png"


def read_mcap_topic_info(mcap_path: Path) -> dict[str, BagTopic]:
    """Read topic metadata from MCAP file by scanning all messages."""
    from mcap.reader import make_reader

    topics: dict[str, BagTopic] = {}

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()

        # Build topic -> schema_id -> schema name mapping
        topic_schema_ids: dict[str, int] = {}
        schema_map: dict[int, str] = {}

        if hasattr(summary, 'channels'):
            for channel in summary.channels.values():
                topic_schema_ids[channel.topic] = channel.schema_id

        if hasattr(summary, 'schemas'):
            for sid, schema in summary.schemas.items():
                schema_map[sid] = schema.name if hasattr(schema, 'name') else "unknown"

        # Count messages per topic
        counts: dict[str, int] = {}
        for (schema, channel, message) in reader.iter_messages():
            topic_name = channel.topic
            counts[topic_name] = counts.get(topic_name, 0) + 1

        for topic_name, count in counts.items():
            schema_id = topic_schema_ids.get(topic_name, 0)
            msg_type = schema_map.get(schema_id, "unknown")
            topics[topic_name] = BagTopic(
                name=topic_name,
                type=msg_type,
                message_count=count
            )

    return topics


def detect_image_type(data: bytes) -> tuple[str, str]:
    """Detect image type from magic bytes. Returns (format_str, image_data).

    JPEG: starts with FF D8 FF
    PNG: starts with 89 50 4E 47
    """
    if len(data) < 4:
        return "unknown", data

    # Check JPEG
    if data[0] == 0xFF and data[1] == 0xD8 and data[2] == 0xFF:
        return "jpeg", data

    # Check PNG
    if data[0] == 0x89 and data[1] == 0x50 and data[2] == 0x4E and data[3] == 0x47:
        return "png", data

    return "unknown", data


def parse_compressed_image_message(data: bytes) -> tuple[int, str, str, bytes]:
    """Parse sensor_msgs/msg/CompressedImage CDR.
    Returns: (timestamp_ns, frame_id, format, image_data).
    """
    try:
        offset = 0

        # Skip header seq (4 bytes)
        offset += 4

        # Time stamp (8 bytes)
        sec = int.from_bytes(data[offset:offset+4], 'little', signed=True)
        nsec = int.from_bytes(data[offset+4:offset+8], 'little')
        timestamp_ns = sec * 1_000_000_000 + nsec
        offset += 8

        # Frame ID string: length (4 bytes) + content
        if offset + 4 > len(data):
            return 0, "", "", b""
        frame_len = int.from_bytes(data[offset:offset+4], 'little')
        offset += 4
        frame_id = data[offset:offset+frame_len].decode('utf-8', errors='ignore')
        offset += frame_len

        # Skip any padding to 4-byte alignment
        while offset % 4 != 0 and offset < len(data):
            offset += 1

        # Now we should be at the format string
        # Format string: length (4 bytes) + content
        if offset + 4 > len(data):
            return timestamp_ns, frame_id, "", b""

        fmt_len = int.from_bytes(data[offset:offset+4], 'little')
        offset += 4
        fmt_str = data[offset:offset+fmt_len].decode('utf-8', errors='ignore')
        offset += fmt_len

        # Skip any padding and null terminator
        while offset % 4 != 0 and offset < len(data):
            if data[offset] == 0:
                offset += 1
            else:
                break

        # Find actual image data by looking for JPEG or PNG magic
        # The data might have CDR encoding issues, so search for magic bytes
        jpeg_magic = bytes([0xFF, 0xD8, 0xFF])
        png_magic = bytes([0x89, 0x50, 0x4E, 0x47])

        img_start = -1
        img_format = "unknown"

        for i in range(offset, min(len(data) - 4, offset + 1000)):
            if data[i:i+3] == jpeg_magic:
                img_start = i
                img_format = "jpeg"
                break
            elif i + 4 <= len(data) and data[i:i+4] == png_magic:
                img_start = i
                img_format = "png"
                break

        if img_start >= 0:
            return timestamp_ns, frame_id, img_format, data[img_start:]

        return timestamp_ns, frame_id, fmt_str, data[offset:]

    except Exception as e:
        return 0, "", "", b""


class MCAPBagReader:
    """Read messages from MCAP ROS2 bag."""

    def __init__(self, mcap_path: Path):
        self.mcap_path = mcap_path
        self._file: BinaryIO | None = None
        self._reader = None

    def __enter__(self):
        self._file = open(self.mcap_path, "rb")
        from mcap.reader import make_reader
        self._reader = make_reader(self._file)
        return self

    def __exit__(self, *args):
        if self._file:
            self._file.close()

    def iter_messages(self):
        """Iterate over all messages yielding (topic, timestamp_ns, data)."""
        if not self._reader:
            raise RuntimeError("Must be used as context manager")

        for (schema, channel, message) in self._reader.iter_messages():
            yield channel.topic, message.publish_time, message.data

    def topics(self) -> dict[str, BagTopic]:
        return read_mcap_topic_info(self.mcap_path)


def detect_image_topics(topics: dict[str, BagTopic]) -> dict[str, str]:
    """Detect which topics are RGB, depth, aligned depth, etc.

    Returns dict mapping topic -> image role.
    """
    color_topics = []
    depth_topics = []
    aligned_depth_topics = []

    for name in topics:
        t = name.lower()
        # Check aligned_depth FIRST to avoid matching "color" in aligned_depth_to_color
        if "aligned" in t and "depth" in t and ("image" in t or "compressed" in t):
            aligned_depth_topics.append(name)
        elif "color" in t and ("image" in t or "compressed" in t):
            color_topics.append(name)
        elif "depth" in t and ("image" in t or "compressed" in t):
            depth_topics.append(name)

    result = {}
    for t in color_topics:
        result[t] = "rgb"
    for t in depth_topics:
        result[t] = "depth"
    for t in aligned_depth_topics:
        result[t] = "aligned_depth"

    return result


def detect_camera_info_topics(topics: dict[str, BagTopic]) -> dict[str, str]:
    """Detect CameraInfo topics."""
    result = {}
    for name in topics:
        t = name.lower()
        if "camera_info" in t or "info" in t:
            if "color" in t:
                result[name] = "rgb_cam_info"
            elif "aligned" in t or "depth" in t:
                result[name] = "depth_cam_info"
    return result


def parse_compressed_image(data: bytes) -> tuple[bytes, str]:
    """Parse CompressedImage and extract raw data and format string."""
    timestamp, frame_id, fmt, img_data = parse_compressed_image_message(data)
    return img_data, fmt