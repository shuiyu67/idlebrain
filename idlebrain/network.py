"""The spiking network itself.

Layout of the neuron array (single flat vector, three contiguous blocks)::

    [ 0 .. n_inputs )                 input layer
    [ n_inputs .. n_inputs+n_hidden ) recurrent ("hidden") layer
    [ ... end )                       output layer, n_classes * group_size

Two properties make this network different from a conventional reservoir:

1. **It runs without input.** About 5% of hidden neurons are pacemakers with
   elevated intrinsic excitability. A slow homeostatic drive pushes the whole
   population toward ``target_hz``, so the network crosses threshold on its own
   after a couple of seconds and then holds a steady rate.

2. **Only the readout is plastic.** Two dense projections learn —
   ``hidden -> output`` and ``input -> output`` — using a three-factor rule::

       Δw = (reward − baseline) × eligibility

   where eligibility counts co-activation within the episode and the baseline
   is a running average of received reward (a reward-prediction error).
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import numpy as np
from scipy import sparse

from .config import Config

__all__ = ["SpikingNetwork"]


class SpikingNetwork:
    """A recurrent spiking network with a plastic readout.

    Parameters
    ----------
    config:
        Structure and dynamics. Defaults are in :class:`~idlebrain.config.Config`.
    seed:
        Seed for connectivity, initial weights and pacemaker placement.

    Examples
    --------
    >>> from idlebrain import Config, SpikingNetwork
    >>> net = SpikingNetwork(Config(n_hidden=500), seed=0)
    >>> net.warmup(steps=2000)                    # doctest: +SKIP
    >>> net.predict(np.zeros(100, dtype=np.float32))   # doctest: +SKIP
    0
    """

    def __init__(self, config: Optional[Config] = None, seed: Optional[int] = None):
        self.config = config or Config()
        self._rng = np.random.default_rng(seed)
        c = self.config

        self.n_neurons = c.n_neurons
        self.input_slice = slice(0, c.n_inputs)
        self.hidden_slice = slice(c.n_inputs, c.n_inputs + c.n_hidden)
        self.output_slice = slice(c.n_inputs + c.n_hidden, c.n_neurons)

        n = self.n_neurons
        self._v = np.zeros(n, np.float32)
        self._adapt = np.zeros(n, np.float32)
        self._resources = np.ones(n, np.float32)
        self._spikes = np.zeros(n, np.float32)
        self._refractory = np.zeros(n, np.int8)

        # Intrinsic excitability: heterogeneous, and higher in a pacemaker
        # minority of the hidden layer. This is what lets it start itself.
        self._bias = self._rng.normal(0.13, 0.19, n).astype(np.float32)
        self._bias[self.input_slice] = 0.0
        self._bias[self.output_slice] = 0.0
        n_pace = max(1, c.n_hidden // 20)
        pace_idx = self._rng.choice(c.n_hidden, n_pace, replace=False)
        self._bias[c.n_inputs + pace_idx] += 0.46

        self._threshold = (1.0 + self._rng.normal(0, 0.08, n)).astype(np.float32)

        self._weights = self._build_recurrent()

        # Plastic readouts.
        self._w_hidden_out = (
            self._rng.normal(0, 1, (c.n_outputs, c.n_hidden))
            * 0.6 / np.sqrt(c.k_rec)
        ).astype(np.float32)
        self._w_input_out = (
            self._rng.normal(0, 1, (c.n_outputs, c.n_inputs))
            * 0.9 / np.sqrt(c.n_inputs)
        ).astype(np.float32)
        self._elig_hidden = np.zeros_like(self._w_hidden_out)
        self._elig_input = np.zeros_like(self._w_input_out)

        # Homeostatic state.
        self._gain = 1.0
        self._drive = 0.0
        self._rate_estimate = 0.0

        # Bookkeeping.
        self._da_baseline = c.da_baseline
        self._spike_total = 0
        self._step_total = 0

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_recurrent(self) -> sparse.csr_matrix:
        c = self.config
        scale_in = 0.55 / np.sqrt(c.k_in)
        scale_rec = 0.50 / np.sqrt(c.k_rec)

        rows, cols, vals = [], [], []

        src = self._rng.integers(0, c.n_inputs, size=c.n_hidden * c.k_in)
        dst = np.repeat(np.arange(c.n_hidden), c.k_in)
        rows.append(dst + c.n_inputs)
        cols.append(src)
        vals.append(self._rng.normal(0, scale_in, dst.size))

        src = self._rng.integers(0, c.n_hidden, size=c.n_hidden * c.k_rec)
        dst = np.repeat(np.arange(c.n_hidden), c.k_rec)
        w = self._rng.normal(0, scale_rec, dst.size)
        inhibitory = self._rng.random(dst.size) < c.p_inhibitory
        w[inhibitory] *= c.w_inhibitory
        rows.append(dst + c.n_inputs)
        cols.append(src + c.n_inputs)
        vals.append(w)

        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_neurons, self.n_neurons),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def n_synapses(self) -> int:
        """Number of stored recurrent synapses."""
        return int(self._weights.nnz)

    @property
    def spikes(self) -> np.ndarray:
        """Boolean-ish vector of which neurons fired on the last step."""
        return self._spikes

    @property
    def rate_hz(self) -> float:
        """Average firing rate since construction, in Hz."""
        if self._step_total == 0:
            return 0.0
        return self._spike_total / self._step_total / self.n_neurons * 1000.0

    @property
    def homeostatic_gain(self) -> float:
        """Current multiplicative gain applied to synaptic input."""
        return self._gain

    @property
    def dopamine_baseline(self) -> float:
        """Running estimate of expected reward."""
        return self._da_baseline

    def hidden_rates(self) -> np.ndarray:
        """Firing rate of each hidden neuron over the last ``predict`` call."""
        return self._hidden_counts / max(1, self._last_steps)

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------

    def step(self, inputs: Optional[np.ndarray] = None) -> int:
        """Advance the simulation by one millisecond.

        Parameters
        ----------
        inputs:
            Optional vector of length ``n_inputs`` in [0, 1]. ``None`` means
            no external stimulation at all — which is the interesting case.

        Returns
        -------
        int
            Number of neurons that fired this step.
        """
        c = self.config

        drive = self._weights @ (self._spikes * self._resources)

        if inputs is not None:
            inp = np.asarray(inputs, dtype=np.float32)
            if inp.shape != (c.n_inputs,):
                raise ValueError(
                    f"inputs must have shape ({c.n_inputs},), got {inp.shape}"
                )
            drive[self.input_slice] += inp * 2.4

        hidden = self._spikes[self.hidden_slice]
        if hidden.any():
            drive[self.output_slice] += (self._w_hidden_out @ hidden) * 1.2
        if inputs is not None:
            drive[self.output_slice] += (self._w_input_out @ inp) * 7.0

        current = (
            drive * self._gain
            + self._bias
            + self._drive
            - c.g_ahp * self._adapt
        )
        self._v += (-self._v + current) / c.tau_m

        self._v[self._refractory > 0] = 0.0
        self._refractory = np.maximum(self._refractory - 1, 0)

        fired = self._v >= self._threshold
        self._v[fired] = 0.0
        self._refractory[fired] = c.refractory_steps
        self._spikes = fired.astype(np.float32)

        # Short-term depression, adaptation, homeostasis.
        self._resources -= c.u_short_term * self._spikes
        np.clip(self._resources, 0.0, None, out=self._resources)
        self._resources += (1.0 - self._resources) / c.tau_recovery
        self._adapt += (self._spikes - self._adapt) / c.tau_ahp

        n_fired = int(fired.sum())
        self._rate_estimate += (n_fired / self.n_neurons - self._rate_estimate) / c.tau_homeostasis
        error = c.target_hz - self._rate_estimate * 1000.0

        self._gain *= np.exp(c.eta_homeostasis * error)
        self._gain = float(np.clip(self._gain, 0.05, 20.0))
        self._drive += c.eta_homeostasis * 1.5 * error
        self._drive = float(np.clip(self._drive, -0.5, 1.8))

        self._spike_total += n_fired
        self._step_total += 1
        return n_fired

    def warmup(self, steps: int = 4000, verbose: bool = False) -> float:
        """Run with no input so the network finds its own level.

        Returns the final average firing rate in Hz.
        """
        for i in range(steps):
            self.step(None)
            if verbose and (i + 1) % 1000 == 0:
                print(f"  step {i + 1:5d}  {self.rate_hz:5.2f} Hz  gain {self._gain:.2f}")
        return self.rate_hz

    def reset(self) -> None:
        """Clear all membrane state, keeping weights and homeostatic settings."""
        self._v.fill(0)
        self._adapt.fill(0)
        self._resources.fill(1)
        self._spikes.fill(0)
        self._refractory.fill(0)
        self._spike_total = 0
        self._step_total = 0

    # ------------------------------------------------------------------
    # readout
    # ------------------------------------------------------------------

    def predict(
        self,
        inputs: np.ndarray,
        steps: int = 100,
        *,
        learn: bool = False,
    ) -> int:
        """Run ``steps`` timesteps with ``inputs`` and return the class.

        If ``learn`` is true, eligibility traces are accumulated as a side
        effect; call :meth:`learn` afterwards with the reward.
        """
        c = self.config
        counts = np.zeros(c.n_outputs, np.int32)
        if learn:
            self._elig_hidden.fill(0)
            self._elig_input.fill(0)
            self._hidden_counts = np.zeros(c.n_hidden)

        for _ in range(steps):
            self.step(inputs)
            out = self._spikes[self.output_slice]
            counts += out.astype(np.int32)
            if learn:
                self._hidden_counts += self._spikes[self.hidden_slice]
                active = np.nonzero(out)[0]
                if active.size:
                    hidden = self._spikes[self.hidden_slice]
                    if hidden.any():
                        self._elig_hidden[np.ix_(active, np.nonzero(hidden)[0])] += 1.0
                    nonzero_in = np.nonzero(inputs)[0]
                    if nonzero_in.size:
                        self._elig_input[np.ix_(active, nonzero_in)] += 1.0

        self._last_steps = steps if learn else getattr(self, "_last_steps", 1)
        votes = counts.reshape(c.n_classes, c.group_size).sum(axis=1)
        return int(np.argmax(votes))

    def train_step(self, inputs: np.ndarray, label: int, steps: int = 100) -> tuple[int, bool]:
        """One supervised episode: predict, reward, update.

        Returns ``(prediction, was_correct)``.
        """
        prediction = self.predict(inputs, steps=steps, learn=True)
        correct = prediction == label
        self.learn(1.0 if correct else 0.0)
        return prediction, correct

    def learn(self, reward: float) -> None:
        """Apply the three-factor update using the last eligibility trace.

        Parameters
        ----------
        reward:
            Scalar reward for the episode just run. The actual learning
            signal is ``reward - baseline``, where the baseline is a running
            average of past rewards.
        """
        c = self.config
        self._da_baseline = 0.97 * self._da_baseline + 0.03 * reward
        delta = reward - self._da_baseline
        if abs(delta) < 1e-7:
            return

        total = float(self._elig_hidden.sum())
        if total > 0:
            scale = 1.0 / max(1.0, total / 200.0)
            mask = self._elig_hidden > 0
            self._w_hidden_out[mask] += (
                c.lr_output * delta * self._elig_hidden[mask] * scale
            )
            self._w_hidden_out *= c.weight_decay

        total_in = float(self._elig_input.sum())
        if total_in > 0:
            scale = 1.0 / max(1.0, total_in / 100.0)
            mask = self._elig_input > 0
            self._w_input_out[mask] += (
                c.lr_output * delta * self._elig_input[mask] * scale * 3.0
            )
            self._w_input_out *= c.weight_decay

        np.clip(self._w_hidden_out, -c.weight_clip, c.weight_clip, out=self._w_hidden_out)
        np.clip(self._w_input_out, -c.weight_clip, c.weight_clip, out=self._w_input_out)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Write weights and config to a single ``.npz`` file."""
        path = pathlib.Path(path)
        np.savez(
            path,
            w_hidden_out=self._w_hidden_out,
            w_input_out=self._w_input_out,
            config_json=np.array(json.dumps(self.config.to_dict())),
            da_baseline=np.array([self._da_baseline], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | pathlib.Path, seed: Optional[int] = None) -> "SpikingNetwork":
        """Restore a network from :meth:`save`.

        The recurrent connectivity is regenerated from ``seed``; only the
        learned readout and configuration are restored. Pass the same seed
        used originally if the hidden layer matters to you.
        """
        with np.load(path, allow_pickle=False) as data:
            config = Config.from_dict(json.loads(str(data["config_json"])))
            net = cls(config, seed=seed)
            if net._w_hidden_out.shape != data["w_hidden_out"].shape:
                raise ValueError(
                    "saved readout shape does not match this config — "
                    "the file was written by a different n_hidden/n_classes/group_size"
                )
            net._w_hidden_out = data["w_hidden_out"].copy()
            net._w_input_out = data["w_input_out"].copy()
            if "da_baseline" in data:
                net._da_baseline = float(data["da_baseline"][0])
        return net

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """One-paragraph human-readable description."""
        c = self.config
        return (
            f"SpikingNetwork(inputs={c.n_inputs}, hidden={c.n_hidden}, "
            f"outputs={c.n_classes}x{c.group_size}={c.n_outputs}, "
            f"synapses={self.n_synapses:,}, rate={self.rate_hz:.2f} Hz)"
        )

    def __repr__(self) -> str:
        return self.summary()

    # lazily created by predict(learn=True)
    _hidden_counts: np.ndarray = np.zeros(0, np.float32)
    _last_steps: int = 1
