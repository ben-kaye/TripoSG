from pathlib import Path

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf

from triposg.ext.briarmbg import BriaRMBG
from triposg.image_utils import prepare_image
from triposg.pipelines.pipeline_triposg import TripoSGPipeline


def string_to_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")
    return dtype


def predictions(cfg: DictConfig):
    rmbg_weights_dir = Path(cfg.rmbg_weights_dir)
    triposg_weights_dir = Path(cfg.triposg_weights_dir)
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)

    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(cfg.device)
    rmbg_net.eval()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_paths = list(Path(cfg.data_dir).rglob("*_rgb.png"))
    # id is stem of image
    with torch.inference_mode():
        images = [
            prepare_image(
                str(im_path), bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net
            )
            for im_path in images_paths
        ]
        # init tripoSG pipeline
        pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(triposg_weights_dir).to(
            device=torch.device(cfg.device), dtype=string_to_torch_dtype(cfg.dtype)
        )

        for k in range(0, len(images), cfg.batch_size):
            images_batch = images[k : k + cfg.batch_size]

            outputs = pipe(
                image=images_batch,
                generator=torch.Generator(device=pipe.device).manual_seed(cfg.seed),
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
            ).samples

            for im_path, output in zip(images_paths[k : k + cfg.batch_size], outputs):
                id_ = im_path.stem.split("_rgb")[0]

                mesh = trimesh.Trimesh(
                    output[0].astype(np.float32), np.ascontiguousarray(output[1])
                )

                file_out = output_dir / f"{id_}.glb"
                mesh.export(file_out)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    args = "config/pipe/toys4k.yaml" if not args else args[0]
    cfg = OmegaConf.load(args)

    predictions(cfg)
