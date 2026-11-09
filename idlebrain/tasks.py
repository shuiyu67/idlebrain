"""Tasks the network can be trained on.

A task is anything that can produce a noisy example of a class and score a
network. Two are bundled: a set of hand-drawn 10x10 patterns, and a generic
wrapper around a user-supplied template array.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Task", "PatternTask", "ArrayTask", "PATTERNS"]

# Ten 10x10 glyphs. '#' is on, everything else off.
PATTERNS: list[list[str]] = [
    ["  ####    ", " ##  ##   ", " ##  ##   ", " ##  ##   ", " ##  ##   ",
     " ##  ##   ", " ##  ##   ", " ##  ##   ", "  ####    ", "          "],
    ["    ##    ", "   ###    ", "  ####    ", "    ##    ", "    ##    ",
     "    ##    ", "    ##    ", "    ##    ", "  ######  ", "          "],
    ["  ####    ", " ##  ##   ", "     ##   ", "    ##    ", "   ##     ",
     "  ##      ", " ##       ", " ##       ", " ######   ", "          "],
    ["  ####    ", " ##  ##   ", "     ##   ", "   ###    ", "     ##   ",
     "     ##   ", "     ##   ", " ##  ##   ", "  ####    ", "          "],
    ["    ##    ", "   ###    ", "  ####    ", " ## ##    ", "##  ##    ",
     "########  ", "    ##    ", "    ##    ", "    ##    ", "          "],
    [" ######   ", " ##       ", " ##       ", " #####    ", "     ##   ",
     "     ##   ", "     ##   ", " ##  ##   ", "  ####    ", "          "],
    ["   ###    ", "  ##      ", " ##       ", " #####    ", " ##  ##   ",
     " ##  ##   ", " ##  ##   ", " ##  ##   ", "  ####    ", "          "],
    [" ######   ", "     ##   ", "    ##    ", "    ##    ", "   ##     ",
     "   ##     ", "  ##      ", "  ##      ", "  ##      ", "          "],
    ["  ####    ", " ##  ##   ", " ##  ##   ", "  ####    ", " ##  ##   ",
     " ##  ##   ", " ##  ##   ", " ##  ##   ", "  ####    ", "          "],
    ["  ####    ", " ##  ##   ", " ##  ##   ", " ##  ##   ", "  #####   ",
     "     ##   ", "     ##   ", "    ##    ", "  ###     ", "          "],
]


class Task(ABC):
    """Base class for anything trainable.

    Subclass it and implement :meth:`sample`. ``n_inputs`` and ``n_classes``
    must match the network's configuration.
    """

    n_inputs: int
    n_classes: int
    noise: float = 0.18

    @abstractmethod
    def sample(self, label: int, rng: np.random.Generator) -> np.ndarray:
        """Return one noisy example of ``label``."""

    def evaluate(
        self,
        network,
        n_per_class: int = 20,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Top-1 accuracy over ``n_per_class`` fresh samples per class."""
        rng = rng or np.random.default_rng()
        correct = total = 0
        for label in range(self.n_classes):
            for _ in range(n_per_class):
                x = self.sample(label, rng)
                correct += network.predict(x) == label
                total += 1
        return correct / max(1, total)

    def confusion(
        self,
        network,
        n_per_class: int = 20,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """``(n_classes, n_classes)`` matrix; row = truth, col = prediction."""
        rng = rng or np.random.default_rng()
        m = np.zeros((self.n_classes, self.n_classes), np.int32)
        for label in range(self.n_classes):
            for _ in range(n_per_class):
                m[label, network.predict(self.sample(label, rng))] += 1
        return m


def _corrupt(x: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Flip each bit independently with probability ``noise``."""
    flip = rng.random(x.shape[0]) < noise
    x = x.copy()
    x[flip] = 1.0 - x[flip]
    return x


class PatternTask(Task):
    """The ten built-in 10x10 glyphs."""

    def __init__(self, noise: float = 0.18, n: int | None = None):
        self.templates = self._render()
        if n is not None:
            self.templates = self.templates[:n]
        self.n_classes = self.templates.shape[0]
        self.n_inputs = self.templates.shape[1]
        self.noise = noise

    @staticmethod
    def _render() -> np.ndarray:
        out = np.zeros((10, 100), np.float32)
        for c, rows in enumerate(PATTERNS):
            for y, row in enumerate(rows):
                for x, ch in enumerate(row):
                    out[c, y * 10 + x] = 1.0 if ch == "#" else 0.0
        return out

    def sample(self, label: int, rng: np.random.Generator) -> np.ndarray:
        return _corrupt(self.templates[label], self.noise, rng)

    def __repr__(self) -> str:
        return f"PatternTask(n_classes={self.n_classes}, noise={self.noise})"


class ArrayTask(Task):
    """Wrap your own templates.

    Parameters
    ----------
    templates:
        ``(n_classes, n_inputs)`` array. Values are compared against 0.5 when
        corrupting, so binary {0,1} data is the expected format.
    """

    def __init__(self, templates: np.ndarray, noise: float = 0.18):
        t = np.asarray(templates, dtype=np.float32)
        if t.ndim != 2:
            raise ValueError("templates must be 2-D (n_classes, n_inputs)")
        self.templates = t
        self.n_classes, self.n_inputs = t.shape
        self.noise = noise

    def sample(self, label: int, rng: np.random.Generator) -> np.ndarray:
        return _corrupt(self.templates[label], self.noise, rng)

    def __repr__(self) -> str:
        return (
            f"ArrayTask(n_classes={self.n_classes}, "
            f"n_inputs={self.n_inputs}, noise={self.noise})"
        )
