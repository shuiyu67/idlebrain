"""Immutable configuration for :class:`idlebrain.SpikingNetwork`.

All parameters that shape the network's structure and dynamics live here.
Nothing in this module has side effects: constructing a ``Config`` never
mutates it, and the network never writes back into it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Config:
    """Structural and dynamical parameters.

    The defaults reproduce the reference experiment in the README:
    2000 hidden neurons reaching ~4.9 Hz spontaneous activity, trained on
    10 classes of 10x10 patterns.

    Attributes
    ----------
    n_inputs:
        Size of the input vector (100 = a 10x10 visual field).
    n_hidden:
        Number of recurrent neurons. This is the only knob that changes
        the network's capacity; everything else is a rate constant.
    n_classes:
        Number of categories to discriminate.
    group_size:
        Output neurons per class. Outputs vote as a population, so a class
        is chosen by summing ``group_size`` spike counts. Larger groups make
        learning more stable at the cost of more output neurons.
    k_in, k_rec:
        Fan-in from the input layer and from the recurrent layer.
    p_inhibitory:
        Fraction of recurrent synapses that are inhibitory.
    """

    # --- structure ---
    n_inputs: int = 100
    n_hidden: int = 2000
    n_classes: int = 10
    group_size: int = 20
    k_in: int = 24
    k_rec: int = 20
    p_inhibitory: float = 0.20
    w_inhibitory: float = -4.0

    # --- neuron dynamics ---
    tau_m: float = 10.0
    v_threshold: float = 1.0
    refractory_steps: int = 3
    tau_ahp: float = 150.0
    g_ahp: float = 0.30
    tau_recovery: float = 120.0
    u_short_term: float = 0.15

    # --- homeostasis ---
    tau_homeostasis: float = 2000.0
    eta_homeostasis: float = 3e-5
    target_hz: float = 5.0

    # --- learning ---
    lr_output: float = 0.02
    da_baseline: float = 0.10
    weight_decay: float = 0.9995
    weight_clip: float = 6.0

    def __post_init__(self) -> None:
        if self.n_inputs < 1:
            raise ValueError("n_inputs must be >= 1")
        if self.n_hidden < 2:
            raise ValueError("n_hidden must be >= 2")
        if self.n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")
        if not (0.0 <= self.p_inhibitory < 1.0):
            raise ValueError("p_inhibitory must be in [0, 1)")
        if self.k_in < 1 or self.k_rec < 1:
            raise ValueError("k_in and k_rec must be >= 1")
        if self.target_hz <= 0:
            raise ValueError("target_hz must be > 0")

    # --- derived quantities ---

    @property
    def n_outputs(self) -> int:
        """Total output neurons (``n_classes * group_size``)."""
        return self.n_classes * self.group_size

    @property
    def n_neurons(self) -> int:
        """Total neurons in the network."""
        return self.n_inputs + self.n_hidden + self.n_outputs

    @property
    def n_synapses(self) -> int:
        """Upper bound on stored recurrent synapses.

        Each hidden neuron is given ``k_in + k_rec`` fan-in slots, but random
        target selection can draw the same (source, destination) pair twice;
        scipy's sparse matrix then merges the duplicates. The network's
        :attr:`SpikingNetwork.n_synapses` reports the de-duplicated count,
        which is always less than or equal to this number.
        """
        return self.n_hidden * (self.k_in + self.k_rec)

    def to_dict(self) -> dict:
        """Plain dict, suitable for serialisation."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Inverse of :meth:`to_dict`. Unknown keys are rejected."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)
