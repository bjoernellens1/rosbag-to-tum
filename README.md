# rosbag-to-tum

Convert ROS bags (MCAP format) to [TUM RGB-D dataset format](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats) with rich metadata.

## Features

- **Auto-detect topics**: RGB, depth, aligned depth, camera info, IMU, odometry, TF
- **Rich metadata**: Camera intrinsics (fx, fy, cx, cy, distortion), extrinsics (camera-to-body), depth format (unit, scale), sensor config
- **Trajectory extraction**: From TF/odometry when available, visual tracker fallback otherwise
- **TUM directory output**: rgb/, depth/, groundtruth.txt, trajectory.txt, intrinsics.yaml, extrinsics.yaml, depth_format.txt
- **Skip existing**: Idempotent runs skip already-converted bags

## Installation

```bash
# Clone
git clone https://github.com/bjoernellens1/rosbag-to-tum.git
cd rosbag-to-tum

# Install with uv
uv sync
```

## Usage

```bash
# Convert all bags in a directory
uv run rosbag-to-tum /path/to/bags --output ./tum_output

# Skip already-converted bags
uv run rosbag-to-tum /path/to/bags --output ./tum_output --skip-existing

# Force visual tracker even if odometry available
uv run rosbag-to-tum /path/to/bags --output ./tum_output --visual-tracker-only

# Filter by bag name
uv run rosbag-to-tum /path/to/bags --output ./tum_output --bag-filter kitchen1
```

## Output Format

```
output/<bag_name>/
├── rgb/                     # RGB images (PNG)
├── depth/                   # Depth images (16-bit PNG, mm precision)
├── groundtruth.txt          # Trajectory from odometry/TF
├── trajectory.txt           # Trajectory from visual tracker (if used)
├── rgb.txt                  # Timestamp -> rgb filename mapping
├── depth.txt                # Timestamp -> depth filename mapping
├── intrinsics.yaml          # Camera intrinsics (fx, fy, cx, cy, distortion)
├── extrinsics.yaml          # Camera-to-base transform
├── depth_format.txt         # Depth unit and scale (NOT TUM's 5000 factor)
├── sensor.yaml              # Sensor type, rates
└── metadata.yaml            # Full bag metadata, topic assignments
```

## Depth Format

Unlike TUM's 5000 factor (where 5000 = 1 meter), this tool preserves the camera's native depth format:

- **RealSense**: millimeters, scale=1.0 (e.g., pixel value 1000 = 1 meter)
- **depth_format.txt** contains the exact unit and scale for downstream tools

## Trajectory

When **odometry/TF is available**: saves to `groundtruth.txt`
When **visual tracker is used**: saves to `trajectory.txt`

Visual tracker uses optical flow + windowed bundle adjustment when no odometry/TF exists.

## Supported Sensors

- Intel RealSense (D400 series)
- Microsoft Kinect
- Azure Kinect

Auto-detected from topic names.

## Bag Directory Structure

The tool expects bags organized as:
```
bags_root/
├── bag_name_a/
│   └── bag_name_a_0.mcap
├── bag_name_b/
│   └── bag_name_b_0.mcap
```

Each `.mcap` file lives in its own folder (the folder name becomes the dataset name).

## Dependencies

- Python 3.11+
- mcap (MCAP bag reading)
- opencv-python (image decode/encode)
- numpy
- pyyaml
- pytransform3d (quaternions)
- click (CLI)
- rich (progress display)