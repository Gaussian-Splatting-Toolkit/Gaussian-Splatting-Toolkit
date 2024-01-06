import sys
import os

# Get the path to the core of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

submodules = [
    "engine",
    "models",
    "evaluation",
    "scene",
    "data",
    "utils",
    "third_party",
    "configs",
    "scripts",
    "surface",
    "viewer",
]

for module in submodules:
    sys.path.append(os.path.join(project_root, module))
    if module == "data":
        sys.path.append(os.path.join(project_root, "data", "dataparsers"))
