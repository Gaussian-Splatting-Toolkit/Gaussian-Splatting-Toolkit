# Surface Distance Evaluation Submodule 📏

The Surface Distance Evaluation submodule within the Gaussian Splatting Toolkit is designed to assess mesh quality by utilizing 3D printing techniques. It facilitates the evaluation of reconstructed meshes against ground truth models, aiding in assessing the accuracy and fidelity of the reconstruction model.

## Table of Contents 📜

- [Surface Distance Evaluation Submodule 📏](#surface-distance-evaluation-submodule-)
  - [Table of Contents 📜](#table-of-contents-)
  - [Introduction 🚀](#introduction-)
  - [Usage 🧰](#usage-)
  - [Dependencies 📦](#dependencies-)
  - [References 🔗](#references-)

## Introduction 🚀

The `surface_distance` submodule offers functionalities to evaluate mesh quality by comparing two files:
- **STL File**: Ground truth file for 3D printing.
- **PLY File**: Generated file from the 3D reconstruction model.

The submodule calculates the error between these files to quantify the accuracy of the reconstruction model, providing valuable insights into mesh quality.

## Usage 🧰

```bash
mkdir build && cd build
cmake ..
make
./surface_distance /path/to/ground_truth.stl /path/to/generated_mesh.ply
```

## Dependencies 📦

This submodule relies on the following external libraries:
- [stlloader](https://github.com/dacunni/stlloader) for loading STL files.
- [happly](https://github.com/nmwsharp/happly) for loading PLY files.
- [TriangleMeshDistance](https://github.com/InteractiveComputerGraphics/TriangleMeshDistance) for calculating distance/error between meshes.

Make sure to install these dependencies before using the submodule.

## References 🔗

- [stlloader](https://github.com/dacunni/stlloader)
- [happly](https://github.com/nmwsharp/happly)
- [TriangleMeshDistance](https://github.com/InteractiveComputerGraphics/TriangleMeshDistance)
