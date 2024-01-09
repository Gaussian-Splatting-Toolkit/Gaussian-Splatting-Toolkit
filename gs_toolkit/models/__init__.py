import sys
import os

# Get the path to the core of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

third_party = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party")
)

diff_gaussian_rasterization_path = os.path.join(
    third_party, "diff-gaussian-rasterization"
)
if diff_gaussian_rasterization_path not in sys.path:
    sys.path.append(diff_gaussian_rasterization_path)

simple_knn_path = os.path.join(third_party, "simple_knn")
if simple_knn_path not in sys.path:
    sys.path.append(simple_knn_path)

submodules = [
    "engine",
    "models",
    "model_components",
    "evaluation",
    "scene",
    "data",
    "utils",
    "configs",
    "scripts",
    "viewer",
]

for module in submodules:
    sys.path.append(os.path.join(project_root, module))
    if module == "data":
        sys.path.append(os.path.join(project_root, "data", "dataparsers"))
