# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/).

## [0.1.0] — 2026-11-09

### Added

- `SpikingNetwork`: three-block LIF network with intrinsic pacemakers,
  homeostatic gain and additive excitation, so it reaches a steady firing rate
  from complete silence.
- Three-factor readout learning (`reward − baseline`) over two plastic
  pathways (hidden→output, input→output).
- Population-coded output (`n_classes × group_size`) with summed spike voting.
- `Config` as an immutable, validated dataclass with derived size properties.
- `Task` abstraction, with `PatternTask` (ten 10×10 glyphs) and `ArrayTask`.
- `Trainer` with evaluation checkpoints and a `History` record.
- Save/load to a single `.npz` (weights + config).
- CLI: `idlebrain` / `python -m idlebrain`.
- 37 unit tests on the standard library only.
- `Trainer.fit(keep_best=True)` restores the best checkpoint automatically.

### Known limitations

- Ten-class accuracy is 35%; the network learns two classes perfectly and
  three nearly so. See README for the measurements.
- The recurrent layer is a fixed random reservoir — no STDP yet.

[0.1.0]: https://github.com/shuiyu67/idlebrain/releases/tag/v0.1.0
