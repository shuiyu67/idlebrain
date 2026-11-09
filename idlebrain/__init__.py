"""idlebrain — spiking networks that wake up on their own, then learn from reward.

Quick start::

    from idlebrain import Config, SpikingNetwork, PatternTask, Trainer

    net = SpikingNetwork(Config(n_hidden=2000), seed=42)
    net.warmup(4000)                      # no input at all: it starts itself
    print(net.rate_hz)                    # ~4.9 Hz

    task = PatternTask(noise=0.18)
    Trainer(net, task).fit(epochs=3000)   # dopamine-driven readout learning

Honest note on what this does and does not do: with a random recurrent layer
and a linear readout, accuracy tracks how linearly separable the classes
already are. Two classes reach 100%, three reach ~99%, ten reach ~35%.
See the README for the measurements.
"""

from .config import Config
from .network import SpikingNetwork
from .tasks import ArrayTask, PatternTask, Task
from .trainer import History, Trainer

__version__ = "0.1.0"

__all__ = [
    "Config",
    "SpikingNetwork",
    "Task",
    "PatternTask",
    "ArrayTask",
    "Trainer",
    "History",
    "__version__",
]
