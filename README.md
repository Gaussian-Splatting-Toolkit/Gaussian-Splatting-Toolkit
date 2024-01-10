# Gaussian Splatting Toolkit 🌐

The Gaussian Splatting Toolkit is a cutting-edge collection of tools designed for new view synthesis using Gaussian splatting techniques, providing a novel explicit 3D representation for scene rendering.

## Table of Contents 📜

- [Gaussian Splatting Toolkit 🌐](#gaussian-splatting-toolkit-)
  - [Table of Contents 📜](#table-of-contents-)
  - [Introduction 🚀](#introduction-)
  - [Features ✨](#features-)
  - [Installation 🛠️](#installation-️)
  - [Usage 🧰](#usage-)
  - [Contribute](#contribute)
  - [TODO](#todo)
  - [License 📝](#license-)

## Introduction 🚀

The Gaussian Splatting Toolkit introduces a groundbreaking technique for new view synthesis utilizing Gaussian splatting. Leveraging little Gaussians to fit the 3D scene, this technique offers a novel explicit 3D representation. It functions as a differentiable render framework capable of updating scene representations end-to-end. Notably, it achieves faster training speeds compared to NeRF (neural radiance field). The project serves as a research tool for further advancements in this domain, especially in robotics applications.

## Features ✨

- **New View Synthesis**: Generates new views of a scene using innovative Gaussian splatting techniques.
- **Novel 3D Representation**: Utilizes little Gaussians to create an explicit and unique 3D scene representation.
- **Differentiable Render Framework**: Enables end-to-end updates of scene representations.
- **Fast Training**: Offers faster training compared to NeRF for scene synthesis and reconstruction.
- **Research Tool**: Designed to facilitate further exploration and research in view synthesis and scene representation.
- **Robotics Application**: Empowers robotics research by swiftly generating digital twins for experimentation and analysis.
- **Additional Modules**: Includes extra modules such as surface extraction for enhanced toolkit functionality.

## Installation 🛠️

To install the Gaussian Splatting Toolkit, follow these steps:

1. Clone the repository: `git clone https://github.com/H-tr/gaussian-splatting-toolkit.git`
2. Navigate to the toolkit directory: `cd gaussian-splatting-toolkit`
3. Install the third party dictionaries
4. Install dependencies: `pip install -e .`

This repository also provides a devcontainer for your convenience.

## Usage 🧰

Here are some basic steps to get started:

1. **New View Synthesis**: Utilize the Gaussian splatting techniques for generating new views of scenes.
2. **Scene Representation**: Explore the explicit 3D representation using little Gaussians.
3. **Differentiable Render Updates**: Experiment with end-to-end scene representation updates.
4. **3D result evaluation**: A set of tools to get the ground truth and calculate the surface distance.
5. **Surface Reconstruction**: Extract the surfaces from 3d Gaussian representation.
6. **Robotics Experimentation**: Use the toolkit for swiftly generating digital twins in robotics research.

For detailed usage instructions, refer to the documentation in each module's directory.

```bash
gs-process-data --input /data/gs-recon/robot_studio.MOV --data-dir /data/gs-recon/rls
```

## Contribute

If you want to add a new submodule, run
```bash
git subtree add --prefix {local directory being pulled into} {remote repo URL} {remote branch} --squash
```

## TODO
- [ ] OpenCV marker ground truth measurement.
- [x] Surface distance module
- [ ] Gaussian Splatting module
- [ ] Camera module
- [ ] Ray module
- [ ] Surface resonstruction module
- [ ] SLAM

## License 📝

This project is licensed under the [MIT License](LICENSE).
