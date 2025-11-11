# improved_viewer.py
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import yaml
from omegaconf import OmegaConf
from PIL import Image

import viser
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class Sample:
    name: str
    glb_path: Path
    image_path: Path


@dataclass(frozen=True)
class DataLoc:
    glb_dir: Path
    image_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Viser viewer for toys4k predictions.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipe/toys4k.yaml"),
        help="Pipeline config to load.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser host.")
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open a browser tab when the server starts.",
    )
    return parser.parse_args()


def load_mesh_data(glb_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(glb_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported GLB contents: {glb_path}")

    if mesh.faces is None or len(mesh.faces) == 0:
        mesh = mesh.convex_hull
    return mesh


def load_image_rgba(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        return np.array(img.convert("RGBA"))


def configure_split_layout(server: viser.ViserServer, fraction: float = 0.5) -> None:
    gui = server.gui
    lock_width = getattr(gui, "lock_width_fraction", None)
    if callable(lock_width):
        lock_width(fraction)
        return

    for attr in ("max_width_fraction", "min_width_fraction", "default_width_fraction"):
        if hasattr(gui, attr):
            try:
                setattr(gui, attr, fraction)
            except (AttributeError, TypeError):
                pass

    add_style = getattr(gui, "add_style", None)
    if callable(add_style):
        percent = int(round(fraction * 100))
        add_style(f":root {{ --viser-gui-width: {percent}vw; }}")


def run_viewer(
    samples: list[Sample],
    host: str,
    port: int,
    open_browser: bool,
    data_loc: DataLoc,
) -> None:
    server = viser.ViserServer(host=host, port=port, open_browser=open_browser)
    configure_split_layout(server, 0.5)

    # --- create stable GUI widgets once and keep references ---
    slider_handle = server.gui.add_slider(
        "Sample index",
        min=0,
        max=max(0, len(samples) - 1),
        step=1,
        initial_value=0,
    )

    # Create a single markdown widget and update its contents later.
    name_display = server.gui.add_markdown("### Sample: —")

    # Create an image widget and keep the handle for updates.
    image_handle = server.gui.add_image(
        np.zeros((1, 1, 4), dtype=np.uint8),
        label="Input image",
    )

    # Keep track of mesh handle and currently displayed sample index
    state: dict[str, object] = {"mesh": None, "current_index": None}

    def _update_or_replace_markdown(handle, text: str):
        """Try to update markdown in-place; if not possible, replace and return new handle."""
        # common update patterns

        if handle is not None:
            handle.content = text
            return handle
        return server.gui.add_markdown(text)

    def _update_or_replace_image(handle, image_data: np.ndarray):
        if handle is None:
            return server.gui.add_image(image_data, label="Input image")
        handle.image = image_data
        return handle

    def _update_mesh_handle(handle, mesh: trimesh.Trimesh):
        """Try to update mesh in place on the handle. If not possible, remove and re-add."""
        if handle is not None:
            handle.remove()
        return server.scene.add_mesh_trimesh(name="prediction", mesh=mesh)

    def set_sample(index: int) -> None:
        nonlocal name_display, image_handle, state
        if not samples:
            return
        index = max(0, min(index, len(samples) - 1))
        if state["current_index"] == index:
            return  # no change
        sample = samples[index]

        # --- mesh update (try in-place first) ---
        mesh = load_mesh_data(sample.glb_path)
        try:
            state["mesh"] = _update_mesh_handle(state.get("mesh"), mesh)
        except Exception as e:
            # last-resort: try to recreate scene or log but don't crash
            print(f"Warning: failed to update mesh handle: {e}")
            try:
                if state.get("mesh") is not None:
                    state["mesh"].remove()
            except Exception:
                pass
            state["mesh"] = server.scene.add_mesh_trimesh(name="prediction", mesh=mesh)

        # --- image update ---
        image_data = load_image_rgba(sample.image_path)
        image_handle = _update_or_replace_image(image_handle, image_data)

        # --- markdown update (update single widget, not append new ones) ---
        name_display = _update_or_replace_markdown(
            name_display, f"### Sample: {sample.name}"
        )

        state["current_index"] = index

    # initial sample
    if samples:
        set_sample(0)

    def slider_cb(*args) -> None:
        nonlocal slider_handle
        set_sample(int(round(slider_handle.value)))

    slider_handle.on_update(slider_cb)

    print(f"Viser running at http://{host}:{port}")

    id_set = set(sample.name for sample in samples)

    try:
        count = 0
        while True:
            # count += 1
            # if not count % 50:
            #     # refresh ids
            #     res = find_ids(data_loc.glb_dir, data_loc.image_dir, id_set)

            #     if res is not None:
            #         logger.warning("Detected new or removed samples; updating viewer.")
            #         samples = res
            #         id_set = set(sample.name for sample in samples)
            #         slider_handle.max = max(0, len(samples) - 1)
            #         # reset to first sample
            #         set_sample(0)

            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down viewer...")
    finally:
        pass


def find_ids(glb_dir: Path, image_dir: Path, id_set: set) -> list[Sample] | None:
    glb_paths = list(glb_dir.glob("*.glb"))
    ids = [p.stem for p in glb_paths]
    if set(ids) == id_set:
        return None

    images_paths = [
        [p for p in image_dir.rglob(f"{id_}*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        for id_ in ids
    ]

    samples = [
        Sample(id_, glb_path, img_list[0])
        for id_, glb_path, img_list in zip(ids, glb_paths, images_paths)
        if img_list
    ]
    return samples


def main() -> None:
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    glb_dir = Path(cfg.output_dir)
    image_dir = Path(cfg.data_dir)

    glb_paths = list(glb_dir.glob("*.glb"))
    ids = [p.stem for p in glb_paths]

    images_paths = [
        [p for p in image_dir.rglob(f"{id_}*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        for id_ in ids
    ]

    samples = [
        Sample(id_, glb_path, img_list[0])
        for id_, glb_path, img_list in zip(ids, glb_paths, images_paths)
        if img_list
    ]

    dataloc = DataLoc(glb_dir=glb_dir, image_dir=image_dir)

    run_viewer(
        samples,
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
        data_loc=dataloc,
    )


if __name__ == "__main__":
    main()
