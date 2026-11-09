"""Command line entry point: ``idlebrain`` or ``python -m idlebrain``."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .config import Config
from .network import SpikingNetwork
from .tasks import PatternTask
from .trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="idlebrain",
        description="Train a self-starting spiking network on the built-in patterns.",
    )
    p.add_argument("--hidden", type=int, default=2000, help="hidden neurons (default: 2000)")
    p.add_argument("--epochs", type=int, default=3000, help="training episodes (default: 3000)")
    p.add_argument("--classes", type=int, default=10, help="how many classes to use (2-10)")
    p.add_argument("--group", type=int, default=20, help="output neurons per class")
    p.add_argument("--noise", type=float, default=0.18, help="pixel flip probability")
    p.add_argument("--seed", type=int, default=20261109, help="random seed")
    p.add_argument("--warmup", type=int, default=4000, help="input-free warmup steps")
    p.add_argument("--save", metavar="PATH", help="write the trained model here")
    p.add_argument("--eval-only", metavar="PATH", help="load a model and just evaluate")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    quiet = args.quiet
    say = (lambda *a, **k: None) if quiet else print

    if not 2 <= args.classes <= 10:
        print("error: --classes must be between 2 and 10", file=sys.stderr)
        return 2

    task = PatternTask(noise=args.noise, n=args.classes)

    if args.eval_only:
        net = SpikingNetwork.load(args.eval_only, seed=args.seed)
        say(net.summary())
        acc = task.evaluate(net, n_per_class=40)
        say(f"accuracy {acc * 100:.1f}%")
        return 0

    cfg = Config(
        n_hidden=args.hidden,
        n_classes=args.classes,
        group_size=args.group,
    )
    say("\n=== idlebrain ===")
    say(f"hidden {cfg.n_hidden}  classes {cfg.n_classes}x{cfg.group_size}  "
        f"synapses {cfg.n_synapses:,}")

    t0 = time.perf_counter()
    net = SpikingNetwork(cfg, seed=args.seed)
    say(f"built in {time.perf_counter() - t0:.2f}s")

    say(f"\nwarmup {args.warmup} steps with no input ...")
    t1 = time.perf_counter()
    net.warmup(args.warmup, verbose=not quiet)
    say(f"  {net.rate_hz:.2f} Hz  gain {net.homeostatic_gain:.2f}  "
        f"{(time.perf_counter() - t1) / args.warmup * 1000:.2f} ms/step")

    say(f"\ntraining {args.epochs} episodes ({args.noise * 100:.0f}% noise)")
    say(f"  {'epoch':>6}  {'accuracy':>9}  {'elapsed':>8}")
    trainer = Trainer(net, task, seed=args.seed + 1)
    history = trainer.fit(args.epochs, eval_every=max(1, args.epochs // 10), verbose=not quiet)

    final = trainer.evaluate(n_per_class=40)
    say(f"\nheld-out accuracy {final * 100:.1f}%  (best seen {history.best_accuracy * 100:.1f}%)")

    seen = task.confusion(net, n_per_class=20)
    say("\nper class:")
    for c in range(task.n_classes):
        hit = seen[c, c] / max(1, seen[c].sum())
        say(f"  {c}: {hit * 100:5.1f}%")

    if args.save:
        net.save(args.save)
        say(f"\nsaved to {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
