"""Training loop with progress reporting and a bit of bookkeeping."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .network import SpikingNetwork
from .tasks import Task

__all__ = ["Trainer", "History"]


@dataclass
class History:
    """Everything that happened during :meth:`Trainer.fit`."""

    epochs: list[int] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    best_accuracy: float = 0.0

    def as_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "accuracy": self.accuracy,
            "seconds": self.seconds,
            "best_accuracy": self.best_accuracy,
        }


class Trainer:
    """Runs reward-modulated episodes on a network.

    Parameters
    ----------
    network:
        The network to train in place.
    task:
        Supplies samples and scores accuracy.
    steps_per_episode:
        How many timesteps each example is presented for.
    seed:
        Seeds the sampling RNG so runs are repeatable.
    """

    def __init__(
        self,
        network: SpikingNetwork,
        task: Task,
        steps_per_episode: int = 100,
        seed: Optional[int] = 0,
    ):
        if task.n_inputs != network.config.n_inputs:
            raise ValueError(
                f"task n_inputs={task.n_inputs} != network n_inputs={network.config.n_inputs}"
            )
        if task.n_classes != network.config.n_classes:
            raise ValueError(
                f"task n_classes={task.n_classes} != network n_classes={network.config.n_classes}"
            )
        self.network = network
        self.task = task
        self.steps_per_episode = steps_per_episode
        self.rng = np.random.default_rng(seed)

    def fit(
        self,
        epochs: int = 5000,
        eval_every: int = 500,
        n_eval: int = 12,
        verbose: bool = True,
        callback: Optional[Callable[[History], None]] = None,
        keep_best: bool = True,
    ) -> History:
        """Train for ``epochs`` episodes.

        Parameters
        ----------
        keep_best:
            Accuracy on this architecture oscillates episode to episode. When
            true (the default), the readout weights from the best evaluation
            checkpoint are restored before returning, so you get the model you
            actually measured rather than whatever the last episode left.

        Returns a :class:`History` with every evaluation checkpoint.
        """
        history = History()
        best_weights: Optional[tuple] = None
        start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            label = int(self.rng.integers(0, self.task.n_classes))
            x = self.task.sample(label, self.rng)
            self.network.train_step(x, label, steps=self.steps_per_episode)

            if epoch % eval_every == 0 or epoch == epochs:
                acc = self.task.evaluate(self.network, n_eval, self.rng)
                elapsed = time.perf_counter() - start
                history.epochs.append(epoch)
                history.accuracy.append(acc)
                history.seconds.append(elapsed)
                if acc > history.best_accuracy:
                    history.best_accuracy = acc
                    if keep_best:
                        best_weights = (
                            self.network._w_hidden_out.copy(),
                            self.network._w_input_out.copy(),
                        )
                if verbose:
                    print(f"  {epoch:6d}  {acc * 100:7.1f}%  {elapsed:7.1f}s", flush=True)
                if callback:
                    callback(history)

        if keep_best and best_weights is not None:
            self.network._w_hidden_out, self.network._w_input_out = best_weights

        return history

    def evaluate(self, n_per_class: int = 40) -> float:
        """Held-out accuracy."""
        return self.task.evaluate(self.network, n_per_class, self.rng)
