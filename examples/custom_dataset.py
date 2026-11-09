"""Bring your own data.

Two things to change: the template array and (optionally) the noise level.
The templates must be ``(n_classes, n_inputs)`` — here we synthesise three
classes from random blobs, but any binary array works, including MNIST rows
thresholded to 0/1 and resized to 10x10.

    python examples/custom_dataset.py
"""

import numpy as np

from idlebrain import ArrayTask, Config, SpikingNetwork, Trainer


def make_blobs(n_classes: int = 3, n_inputs: int = 100, seed: int = 0) -> np.ndarray:
    """Three 10x10 templates, each with a bright square in a different corner."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n_classes, n_inputs), np.float32)
    side = 10
    size = 4
    for c in range(n_classes):
        oy = (c // 2) * (side - size)
        ox = (c % 2) * (side - size)
        for y in range(oy, oy + size):
            for x in range(ox, ox + size):
                out[c, y * side + x] = 1.0
        out[c] = np.clip(out[c] + rng.normal(0, 0.02, n_inputs), 0, 1)
    return out


templates = make_blobs()
print(f"templates: {templates.shape}, density {templates.mean():.2f}")

task = ArrayTask(templates, noise=0.12)
net = SpikingNetwork(
    Config(
        n_inputs=task.n_inputs,
        n_hidden=800,
        n_classes=task.n_classes,
        group_size=15,
    ),
    seed=7,
)
net.warmup(3000)
print(f"spontaneous rate before training: {net.rate_hz:.2f} Hz")

trainer = Trainer(net, task, seed=3)
trainer.fit(epochs=2000, eval_every=500)
print(f"\nfinal accuracy {trainer.evaluate(30) * 100:.0f}%")

net.save("custom_model.npz")
print("saved to custom_model.npz")

reloaded = SpikingNetwork.load("custom_model.npz", seed=7)
print(f"reloaded model scores {task.evaluate(reloaded, 30) * 100:.0f}%")
