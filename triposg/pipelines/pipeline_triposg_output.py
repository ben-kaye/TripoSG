from dataclasses import dataclass

import torch
import trimesh
from diffusers.utils import BaseOutput


@dataclass
class TripoSGPipelineOutput(BaseOutput):
    r"""
    Output class for ShapeDiff pipelines.
    """

    samples: torch.Tensor
    meshes: list[trimesh.Trimesh]
