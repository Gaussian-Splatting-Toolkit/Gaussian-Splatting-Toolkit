from os import path
from pathlib import Path
from typing import Optional

import appdirs
import torch
from torch.utils.data import DataLoader
import numpy as np
import requests
from rich.progress import track

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.demo_utils import flush_buffer
from deva.ext.grounding_dino import get_grounding_dino_model
from deva.ext.with_text_processor import process_frame_with_text as process_frame
from deva.model.network import DEVA
from gs_toolkit.utils.rich_utils import CONSOLE, status

import json


def generate_mask_from_text(
    model: str = "DEVA-propagation.pth",
    output: Optional[Path] = None,
    save_all: bool = False,
    size: int = 480,
    amp: bool = True,
    key_dim: int = 64,
    value_dim: int = 512,
    pix_feat_dim: int = 512,
    disable_long_term: bool = False,
    max_mid_term_frames: int = 10,
    min_mid_term_frames: int = 5,
    max_long_term_elements: int = 10000,
    num_prototypes: int = 128,
    top_k: int = 30,
    mem_every: int = 5,
    chunk_size: int = 4,
    GROUNDING_DINO_CONFIG_NAME: str = "GroundingDINO_SwinT_OGC.py",
    GROUNDING_DINO_CHECKPOINT_NAME: str = "groundingdino_swint_ogc.pth",
    DINO_THRESHOLD: float = 0.35,
    DINO_NMS_THRESHOLD: float = 0.8,
    SAM_ENCODER_VERSION: str = "vit_h",
    SAM_CHECKPOINT_NAME: str = "sam_vit_h_4b8939.pth",
    MOBILE_SAM_CHECKPOINT_NAME: str = "mobile_sam.pt",
    SAM_NUM_POINTS_PER_SIDE: int = 64,
    SAM_NUM_POINTS_PER_BATCH: int = 64,
    SAM_PRED_IOU_THRESHOLD: float = 0.88,
    SAM_OVERLAP_THRESHOLD: float = 0.8,
    img_path: Optional[Path] = None,
    detection_every: int = 5,
    num_voting_frames: int = 3,
    temporal_setting: str = "semionline",
    max_missed_detection_count: int = 10,
    max_num_objects: int = -1,
    prompt: Optional[str] = None,
    sam_variant: str = "original",
    enable_long_term_count_usage: bool = False,
    verbose: bool = False,
):
    ckpt_dir = Path(appdirs.user_data_dir("gs_toolkit"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = ckpt_dir / model

    GROUNDING_DINO_CONFIG_PATH = ckpt_dir / GROUNDING_DINO_CONFIG_NAME
    GROUNDING_DINO_CHECKPOINT_PATH = ckpt_dir / GROUNDING_DINO_CHECKPOINT_NAME
    SAM_CHECKPOINT_PATH = ckpt_dir / SAM_CHECKPOINT_NAME
    MOBILE_SAM_CHECKPOINT_PATH = ckpt_dir / MOBILE_SAM_CHECKPOINT_NAME

    urls_dict = {
        model: "https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth",
        GROUNDING_DINO_CONFIG_PATH: "https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/GroundingDINO_SwinT_OGC.py",
        GROUNDING_DINO_CHECKPOINT_PATH: "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        SAM_CHECKPOINT_PATH: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        MOBILE_SAM_CHECKPOINT_PATH: "https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/mobile_sam.pt",
    }

    for file_path, url in urls_dict.items():
        if not file_path.exists():
            r = requests.get(url, stream=True)
            with open(file_path, "wb") as f:
                total_length = int(r.headers.get("content-length"))
                if file_path == GROUNDING_DINO_CONFIG_PATH:
                    total_length /= 2
                assert total_length is not None
                for chunk in track(
                    r.iter_content(chunk_size=1024),
                    total=(total_length / 1024) + 1,
                    description=f"Downloading {file_path.name}...",
                ):
                    if chunk:
                        f.write(chunk)
                        f.flush()

    torch.autograd.set_grad_enabled(False)
    np.random.seed(42)

    """
    Arguments loading
    """

    config = {
        "model": model,
        "output": str(output),
        "save_all": save_all,
        "size": size,
        "amp": amp,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "pix_feat_dim": pix_feat_dim,
        "disable_long_term": disable_long_term,
        "max_mid_term_frames": max_mid_term_frames,
        "min_mid_term_frames": min_mid_term_frames,
        "max_long_term_elements": max_long_term_elements,
        "num_prototypes": num_prototypes,
        "top_k": top_k,
        "mem_every": mem_every,
        "chunk_size": chunk_size,
        "GROUNDING_DINO_CONFIG_PATH": GROUNDING_DINO_CONFIG_PATH,
        "GROUNDING_DINO_CHECKPOINT_PATH": GROUNDING_DINO_CHECKPOINT_PATH,
        "DINO_THRESHOLD": DINO_THRESHOLD,
        "DINO_NMS_THRESHOLD": DINO_NMS_THRESHOLD,
        "SAM_ENCODER_VERSION": SAM_ENCODER_VERSION,
        "SAM_CHECKPOINT_PATH": SAM_CHECKPOINT_PATH,
        "MOBILE_SAM_CHECKPOINT_PATH": MOBILE_SAM_CHECKPOINT_PATH,
        "SAM_NUM_POINTS_PER_SIDE": SAM_NUM_POINTS_PER_SIDE,
        "SAM_NUM_POINTS_PER_BATCH": SAM_NUM_POINTS_PER_BATCH,
        "SAM_PRED_IOU_THRESHOLD": SAM_PRED_IOU_THRESHOLD,
        "SAM_OVERLAP_THRESHOLD": SAM_OVERLAP_THRESHOLD,
        "img_path": str(img_path),
        "detection_every": detection_every,
        "num_voting_frames": num_voting_frames,
        "temporal_setting": temporal_setting,
        "max_missed_detection_count": max_missed_detection_count,
        "max_num_objects": max_num_objects,
        "prompt": prompt,
        "sam_variant": sam_variant,
        "enable_long_term": not disable_long_term,
        "enable_long_term_count_usage": enable_long_term_count_usage,
    }

    deva_model = DEVA(config).cuda().eval()
    if config["model"] is not None:
        model_weights = torch.load(config["model"])
        deva_model.load_weights(model_weights)
    gd_model, sam_model = get_grounding_dino_model(config, "cuda")
    """
    Temporal setting
    """
    config["temporal_setting"] = config["temporal_setting"].lower()
    assert config["temporal_setting"] in ["semionline", "online"]

    # get data
    video_reader = SimpleVideoReader(config["img_path"])
    loader = DataLoader(
        video_reader, batch_size=None, collate_fn=no_collate, num_workers=8
    )
    out_path = config["output"]

    # Start eval
    vid_length = len(loader)
    # no need to count usage for LT if the video is not that long anyway
    config["enable_long_term_count_usage"] = (
        config["enable_long_term"]
        and (
            vid_length
            / (config["max_mid_term_frames"] - config["min_mid_term_frames"])
            * config["num_prototypes"]
        )
        >= config["max_long_term_elements"]
    )

    CONSOLE.print("[bold yellow]Mask Generation Configuration:", config)

    deva = DEVAInferenceCore(deva_model, config=config)
    deva.next_voting_frame = config["num_voting_frames"] - 1
    deva.enabled_long_id()
    result_saver = ResultSaver(
        out_path, None, dataset="demo", object_manager=deva.object_manager
    )

    with status(
        msg="[bold yellow]Generating Masks",
        spinner="circle",
        verbose=verbose,
    ):
        with torch.cuda.amp.autocast(enabled=config["amp"]):
            for ti, (frame, im_path) in enumerate(loader):
                process_frame(
                    deva, gd_model, sam_model, im_path, result_saver, ti, image_np=frame
                )
            flush_buffer(deva, result_saver)
        result_saver.end()
    CONSOLE.print(
        f"[bold green]:white_check_mark: Done! Masks saved at {out_path}[/bold green]"
    )

    # save this as a video-level json
    with open(path.join(out_path, "pred.json"), "w") as f:
        json.dump(result_saver.video_json, f, indent=4)  # prettier json
