"""Run examples with their default settings, writing results/: python examples/run_all.py [name ...].

All of them take a few hours (the shape reconstructions are the longest).
"""

import json
import sys
import time

import appearance
import env_map
import isolated
import moving_light
import multipose
import neural_texture
import optimize_scene
import quickstart
import shape
import texture
import translation

EXAMPLES = {
    "quickstart": quickstart.run,
    "materials": lambda: appearance.run("materials"),
    "lights": lambda: appearance.run("lights"),
    "translation": translation.run,
    "shape": shape.run,
    "texture": texture.run,
    "moving_light": moving_light.run,
    "env_map": env_map.run,
    "neural_texture": neural_texture.run,
    "optimize_scene": optimize_scene.run,
    "multipose": multipose.run,
    "shadow_only": lambda: isolated.run("shadow"),
    "mirror_only": lambda: isolated.run("mirror"),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(EXAMPLES)
    unknown = [name for name in names if name not in EXAMPLES]
    if unknown:
        sys.exit(f"unknown examples: {' '.join(unknown)} (choose from {' '.join(EXAMPLES)})")
    for name in names:
        start = time.perf_counter()
        metrics = EXAMPLES[name]()
        metrics["seconds"] = time.perf_counter() - start
        print(name, json.dumps(metrics), flush=True)
