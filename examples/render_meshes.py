"""Render optimized meshes (mesh.npz of the shape-reconstruction examples) next to their target from several
directions: python examples/render_meshes.py results/isolated_shadow_cube ... [--out results/meshes.png]."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from falcor2_inverse.images import frame, stack
from common import RESULTS, SHAPES, MeshViewer, target_shape

DIRECTIONS = [(0, -1, 0.2), (1, -0.6, 0.3), (-1, -0.3, 0.6), (0, 1, 0.3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--out", default=str(RESULTS / "meshes.png"))
    args = parser.parse_args()
    viewers = [MeshViewer(direction=d) for d in DIRECTIONS]
    rows, targets = [], set()
    for path in args.results:
        shape = next(s for s in SHAPES if s in Path(path).name.split("_"))
        if shape not in targets:
            targets.add(shape)
            mesh = target_shape(shape)
            rows.append(frame([v.render(*mesh) for v in viewers], [f"true {shape}"] + [""] * (len(viewers) - 1)))
        mesh = np.load(Path(path) / "mesh.npz")
        history = json.loads((Path(path) / "history.json").read_text())
        label = f"{Path(path).name}: chamfer {history['final_chamfer']:.4f}"
        if "intersecting_edges" in history:
            label += f", {history['intersecting_edges']} crossing edges"
        images = [v.render(mesh["positions"], mesh["faces"]) for v in viewers]
        rows.append(frame(images, [label] + [""] * (len(viewers) - 1)))
    Image.fromarray(stack(rows)).save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
