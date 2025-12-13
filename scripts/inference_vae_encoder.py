from pathlib import Path

import kaolin
import numpy as np
import torch

from triposg.models.autoencoders import TripoSGVAEModel
from triposg.pipelines.pipeline_vae_encoder import TripoSGEncoderPipeline


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE model
    # TODO: Replace with actual model loading path or from_pretrained
    # Assuming the model is available at a certain path or HuggingFace repo
    # For example:
    # vae = TripoSGVAEModel.from_pretrained("path/to/model")
    # Since latent_channels=64 by default, it should match
    vae = TripoSGVAEModel()  # Placeholder, replace with actual loading
    vae.to(device)

    # Create pipeline
    pipeline = TripoSGEncoderPipeline(vae)
    pipeline.to(device)

    # Dataset path
    dataset_path = Path("/datasets/ycb2")

    # Loop through objects
    for object_dir in dataset_path.iterdir():
        if not object_dir.is_dir():
            continue

        obj_path = object_dir / "processed" / "collision.obj"
        if not obj_path.exists():
            print(f"Skipping {object_dir.name}: collision.obj not found")
            continue

        print(f"Processing {object_dir.name}")

        # Load mesh
        mesh = kaolin.io.obj.import_mesh(str(obj_path))

        # Encode to latents
        latents = pipeline(mesh.vertices, mesh.faces).squeeze(0)

        # Save latents
        latents_path = object_dir / "latents.npy"
        np.save(str(latents_path), latents.cpu().numpy())

        print(f"Saved latents to {latents_path}")


if __name__ == "__main__":
    main()
