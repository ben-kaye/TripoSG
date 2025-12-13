import kaolin
import numpy as np
import torch
from diffusers.models.modeling_outputs import AutoencoderKLOutput as KLOutput
from diffusers.utils import logging

# Keep relevant model imports
from ..models.autoencoders import TripoSGVAEModel

logger = logging.get_logger(__name__)


class TripoSGEncoderPipeline:
    """
    Pipeline specifically for Mesh-to-Latent encoding using the TripoSG VAE
    and Kaolin for geometry sampling.
    """

    def __init__(
        self,
        vae: TripoSGVAEModel,
    ):
        super().__init__()
        self.vae = vae
        self.device = next(self.vae.parameters()).device

    @torch.no_grad()
    def __call__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        num_tokens: int = 64,
        num_samples: int = 20480,  # Typical default for high-res
        generator: torch.Generator | None = None,
    ):
        """
        Encodes a mesh (vertices and faces) into VAE latents.

        Args:
            vertices (torch.Tensor): Tensor of shape (V, 3).
            faces (torch.Tensor): Tensor of shape (F, 3) (long).
            num_samples (int): Number of points to sample from the surface.

        Returns:
            torch.Tensor: The encoded latents.
        """
        self.vae.eval()

        # 1. Prepare Data with Kaolin
        # Ensure inputs are on the correct device and have batch dim if needed
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)  # (1, V, 3)
        if faces.dim() == 2:
            # Kaolin usually expects faces to be on the same device
            faces = faces.to(self.device)

        vertices = vertices.to(self.device)

        # Sample points on the mesh surface
        # points: (Batch, num_samples, 3)
        # face_indices: (Batch, num_samples) - telling us which face each point came from
        points, face_indices = kaolin.ops.mesh.sample_points(
            vertices, faces, num_samples=num_samples
        )

        # 2. Compute Normals using Kaolin
        # First, get the vertices corresponding to each face: (Batch, F, 3, 3)
        face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices, faces)

        # Calculate face normals: (Batch, F, 3)
        # unit=True ensures they are normalized
        mesh_face_normals = kaolin.ops.mesh.face_normals(face_vertices, unit=True)

        # Gather the normals for the sampled points based on which face they land on
        # We need to expand face_indices to gather along the 3 normal dimensions
        # face_indices is (B, N), we need indices for (B, N, 3)
        batch_indices = torch.arange(vertices.shape[0], device=self.device).view(
            -1, 1, 1
        )
        face_indices_expanded = face_indices.unsqueeze(-1).expand(-1, -1, 3)

        # Gather: (Batch, num_samples, 3)
        sampled_normals = torch.gather(mesh_face_normals, 1, face_indices_expanded)

        # 3. Format Input for VAE
        # Standard implicit VAEs often expect concatenated Points + Normals
        # Shape: (Batch, Num_Samples, 6) -> [x, y, z, nx, ny, nz]
        vae_input = torch.cat([points, sampled_normals], dim=-1)

        # 4. Encode
        # Note: Depending on the specific TripoSGVAE signature, you might need to
        # transpose this to (Batch, Channels, N) or pass it as is.
        # Assuming standard (B, N, C) or (B, C, N).
        # If the model expects channels first:
        # vae_input = vae_input.transpose(1, 2)

        posterior = self.vae.encode(vae_input, num_tokens=num_tokens)

        if isinstance(posterior, KLOutput):
            posterior = posterior.latent_dist

        # Sample from posterior to get the actual latents
        latents = (
            posterior.sample(generator=generator)
            if hasattr(posterior, "sample")
            else posterior
        )

        return latents

    def to(self, device: str | torch.device):
        self.vae.to(device)
        self.device = device
        return self
