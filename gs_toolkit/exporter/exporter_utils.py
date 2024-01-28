from typing import Any, Dict, List, Tuple, Optional
from gs_toolkit.pipelines.base_pipeline import VanillaPipeline
from gs_toolkit.data.datasets.base_dataset import InputDataset

def collect_camera_poses_for_dataset(dataset: Optional[InputDataset]) -> List[Dict[str, Any]]:
    """Collects rescaled, translated and optimised camera poses for a dataset.

    Args:
        dataset: Dataset to collect camera poses for.

    Returns:
        List of dicts containing camera poses.
    """

    if dataset is None:
        return []

    cameras = dataset.cameras
    image_filenames = dataset.image_filenames

    frames: List[Dict[str, Any]] = []

    # new cameras are in cameras, whereas image paths are stored in a private member of the dataset
    for idx in range(len(cameras)):
        image_filename = image_filenames[idx]
        transform = cameras.camera_to_worlds[idx].tolist()
        frames.append(
            {
                "file_path": str(image_filename),
                "transform": transform,
            }
        )

    return frames

def collect_camera_poses(pipeline: VanillaPipeline) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collects camera poses for train and eval datasets.

    Args:
        pipeline: Pipeline to evaluate with.

    Returns:
        List of train camera poses, list of eval camera poses.
    """

    train_dataset = pipeline.datamanager.train_dataset
    assert isinstance(train_dataset, InputDataset)

    eval_dataset = pipeline.datamanager.eval_dataset
    assert isinstance(eval_dataset, InputDataset)

    train_frames = collect_camera_poses_for_dataset(train_dataset)
    eval_frames = collect_camera_poses_for_dataset(eval_dataset)

    return train_frames, eval_frames