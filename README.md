# Gaussian Splatting Toolkit

The Gaussian Splatting Toolkit is a cutting-edge collection of tools designed for new view synthesis using Gaussian splatting techniques, providing a novel explicit 3D representation for scene rendering.

## Table of Contents

- [Gaussian Splatting Toolkit](#gaussian-splatting-toolkit)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Download the opensource datasets](#download-the-opensource-datasets)
    - [Data processing](#data-processing)
    - [Train the Gaussian Splatting](#train-the-gaussian-splatting)
    - [Visualize the result](#visualize-the-result)
    - [Render the rgb and depth](#render-the-rgb-and-depth)
    - [Export](#export)
  - [Contribute](#contribute)
  - [TODO](#todo)

## Introduction

A Gaussian Splatting Toolkit for robotics research. Developed based on nerfstudio.

## Installation

To install the Gaussian Splatting Toolkit, follow these steps:

1. Clone the repository: `git clone https://github.com/H-tr/gaussian-splatting-toolkit.git --recursive`
2. Navigate to the toolkit directory: `cd gaussian-splatting-toolkit`
3. Install the third party dictionaries
4. Install dependencies: `pip install -e .`

Using conda:

```bash
conda create -n gstk python=3.10.13 -y
conda activate gstk
pip install torch torchvision
pip install -e .
```

This repository also provides a devcontainer for your convenience.

## Usage

### Download the opensource datasets

```bash
gs-download-data gstk --save-dir /path/to/save/dir --capture-name all
```

### Data processing

```bash
# Extract from video
gs-process-data video --data /path/to/video --output-dir /path/to/output-dir --num-frames-target 1000
# Extract from images
gs-process-data images --data /path/to/image/folder --output-dir /path/to/output-dir
# Extract with both rgb and depth
gs-process-data images --data /path/to/rgb/folder --depth-data /path/to/depth/folder --output-dir /path/to/output-dir
```

### Train the Gaussian Splatting

```bash
gs-train gaussian-splatting --data /path/to/processed/data
gs-train depth-gs --data /path/to/processed/data
```

### Visualize the result

```bash
gs-viewer --load-config outputs/path/to/config.yml
```

### Render the rgb and depth

From trajectory

```bash
gs-render trajectory --trajectory-path /path/to/trajectory.json --config-file /path/to/ckpt/config.yml
```

From camera pose

```bash
gs-render pose --config-file /path/to/config.yml --output-dir /path/to/output/folder/
```

### Export

Export the gaussians as ply

```bash
gs-export gaussian-splat --load-config /path/to/config.yml --output-dir exports/gaussians/
```

Export camera poses

```bash
gs-export camera-poses --load-config /path/to/config.yml --output-dir exports/cameras/
```

Export the point cloud

```bash
gs-export point-cloud --load-config /path/to/config.yml --output-dir exports/pcd/
```

```bash
gs-export offline-tsdf --render-path /path/to/rendered/folder --output-dir exports/tsdf/
```

## Contribute

To add a new submodule, run

```bash
git subtree add --prefix {local directory being pulled into} {remote repo URL} {remote branch} --squash
```

## TODO

- [x] OpenCV marker ground truth measurement.
- [x] Surface distance module
- [ ] Data
  - [ ] Data synthetic with blender
- [ ] Data preprocessing
  - [x] Colmap preprocessing
  - [x] RGB-D Data processing
  - [ ] Sensor interface
    - [ ] Azure Kinect
    - [ ] iPhone/ iPad
- [x] Evaluation
- [x] Gaussian Splatting module
- [x] Depth Loss
- [x] Point cloud export
- [x] Mesh extraction
  - [x] Marching cube
  - [x] TSDF
  - [x] Piossion reconstruction
- [x] Training
  - [x] Course to fine
- [ ] Mask
- [ ] Model
  - [ ] Gaussian Splatting SLAM
  - [ ] GaussianShader
- [ ] Visualization
  - [ ] normal visualization
  - [ ] gaussian ellipsoid visualization
- [x] Render
  - [x] Render GS model without loading pipeline
- [ ] Documentation
- [ ] Tests
- [x] CUDA
  - [x] migrate the rasterizer to cuda-12.1
