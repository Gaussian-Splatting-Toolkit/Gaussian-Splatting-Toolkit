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

This repository also provides a devcontainer for your convenience.

## Usage

```bash
gs-process-data video --data /data/gs-recon/robot_studio.MOV --output-dir /data/gs-recon/robot_studio --sfm-tool colmap --num-frames-target 1000
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
- [ ] Depth Loss
- [ ] Point cloud export
- [ ] Mesh extraction
  - [ ] Marching cube
  - [ ] TSDF
  - [ ] Piossion reconstruction
- [ ] Model
  - [ ] SuGaR
  - [ ] SplaTAM
- [x] Visualization
- [ ] Documentation
- [ ] Tests
