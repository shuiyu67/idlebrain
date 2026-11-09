"""Minimal example: watch the network start itself, then teach it two classes.

    python examples/quickstart.py
"""

import numpy as np

from idlebrain import Config, PatternTask, SpikingNetwork, Trainer

# --- 1. build it -------------------------------------------------------
# 500 hidden neurons is enough to see the behaviour and finishes fast.
config = Config(n_hidden=500, n_classes=2, group_size=10)
net = SpikingNetwork(config, seed=42)
print(net.summary())

# --- 2. switch everything off ------------------------------------------
# warmup() calls step(None): no input, no clock beyond the timestep itself.
# The only thing driving activity is intrinsic excitability + homeostasis.
print("\nno input at all:")
for block in range(10):
    for _ in range(200):
        net.step(None)
    print(f"  {200 * (block + 1):5d} steps   {net.rate_hz:5.2f} Hz")

# --- 3. teach it two classes -------------------------------------------
task = PatternTask(noise=0.15, n=2)
trainer = Trainer(net, task, seed=1)

print(f"\ntraining on {task.n_classes} classes:")
history = trainer.fit(epochs=1200, eval_every=300)

print(f"\naccuracy {trainer.evaluate(20) * 100:.0f}%  "
      f"(random would be {100 / task.n_classes:.0f}%)")

# --- 4. look at what it actually says ----------------------------------
print("\nper class:")
m = task.confusion(net, n_per_class=20)
for c in range(task.n_classes):
    print(f"  {c}: {m[c, c]}/{m[c].sum()}")
