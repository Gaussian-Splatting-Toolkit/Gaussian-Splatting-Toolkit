# Gaussian Splatting Toolkit

The Gaussian Splatting Toolkit is a cutting-edge collection of tools designed for new view synthesis using Gaussian splatting techniques, providing a novel explicit 3D representation for scene rendering.

## Table of Contents

- [Gaussian Splatting Toolkit](#gaussian-splatting-toolkit)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contribute](#contribute)
  - [TODO](#todo)

## Introduction

A Gaussian Splatting Toolkit for robotics research. Developed based on nerfstudio.

## Installation

To install the Gaussian Splatting Toolkit, follow these steps:

1. Clone the repository: `git clone https://github.com/H-tr/gaussian-splatting-toolkit.git`
2. Navigate to the toolkit directory: `cd gaussian-splatting-toolkit`
3. Install the third party dictionaries
4. Install dependencies: `pip install -e .`

If you are a developer, and want to install the dev dependencies, please run

```bash
poetry install
```

or

```bash
pip install -r requirement_dev.txt
```

This repository also provides a devcontainer for your convenience.

## Usage

```bash
# Extract from video
gs-process-data video --data /data/gs-recon/robot_studio.MOV --output-dir /data/gs-recon/robot_studio --num-frames-target 1000
# Extract from images
gs-process-data images --data /data/depth_scan/half_rls/rgb --output-dir /data/gs-recon/half_rls_rgb
# Extract with both rgb and depth
gs-process-data images --data /data/depth_scan/half_rls/rgb --depth-data /data/depth_scan/half_rls/depth --output-dir /data/gs-recon/half_rls_full
```

```bash
gs-train --data /data/gs-recon/robot_studio
```

## Contribute

To add a new submodule, run
```bash
git subtree add --prefix {local directory being pulled into} {remote repo URL} {remote branch} --squash
```

## TODO
- [x] OpenCV marker ground truth measurement.
- [x] Surface distance module
- [ ] Data preprocessing
  - [x] Colmap preprocessing
  - [ ] RGB-D Data processing
  - [ ] Sensor interface
    - [ ] Azure Kinect
    - [ ] iPhone/ iPad
- [ ] Benchmarking
- [x] Gaussian Splatting module
- [ ] Gaussian Refinement
- [x] Depth Loss
- [x] Point cloud export
- [ ] Mesh extraction
  - [ ] Marching cube
  - [ ] TSDF
  - [ ] Piossion reconstruction
- [ ] Model
  - [ ] SuGaR
  - [ ] SplaTAM
- [x] Visualization
- [ ] Render
  - [ ] Render GS model without loading pipeline
- [ ] Documentation
- [ ] Tests
