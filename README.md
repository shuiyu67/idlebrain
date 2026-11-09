# idlebrain

Spiking neural networks that **wake up on their own**, then learn from dopamine.

No input. No external clock. No pretrained weights. Just neurons with slightly
different excitability, a slow homeostatic drive, and a rule that says *do that
again* when something goes right.

```python
from idlebrain import Config, PatternTask, SpikingNetwork, Trainer

net = SpikingNetwork(Config(n_hidden=2000), seed=42)
net.warmup(4000)                        # nothing is fed in
print(net.rate_hz)                      # ~4.9 Hz — it started itself

task = PatternTask(noise=0.18)
Trainer(net, task).fit(epochs=3000)     # reward-modulated readout
```

Pure NumPy + `scipy.sparse`. No deep learning framework. Runs on CPU.

---

## Install

```bash
pip install idlebrain
```

From source:

```bash
git clone https://github.com/shuiyu67/idlebrain
cd idlebrain
pip install -e .
```

China mirror (Gitee):

```bash
git clone https://gitee.com/shuiyu1123/idlebrain
```

Requires Python ≥ 3.9, NumPy ≥ 1.22, SciPy ≥ 1.8.

---

## What it actually does

The network has three blocks of neurons:

```
   input (100)
      │        ╲
      ↓         ╲
   hidden (N) ───→ output (n_classes × group_size)
      ↑                 ↑
      └── recurrent ────┘ (plastic)
```

**The hidden layer runs by itself.** About 5% of its neurons have elevated
intrinsic excitability. A homeostatic term nudges the whole population toward
`target_hz`. Start from silence and it crosses threshold after a couple of
seconds, overshoots, then settles:

```
0.1 → 0.3 → 1.0 → 2.4 → 4.0 → 4.8 → 4.9 Hz
```

**Only the readout learns.** Two dense projections — `hidden → output` and
`input → output` — are updated by:

```
Δw = (reward − baseline) × eligibility
```

where eligibility counts how often a source and its output fired together in
the episode, and the baseline is a running average of reward (a reward
prediction error, not a fixed constant).

---

## Measured accuracy

2000 hidden neurons, 18% pixel noise, 4000 episodes, 40 held-out samples per class:

| classes | accuracy | chance |
|--------:|---------:|-------:|
| **2**   | **100.0%** | 50% |
| **3**   | **98.9%**  | 33% |
| 5       | 65.3%      | 20% |
| 10      | 35.0%      | 10% |

**Read this before adopting it.** The random recurrent layer destroys linear
separability — measured directly: raw pixels are 99.0% linearly separable,
the hidden layer's firing rates are **0.9%**. So the readout ends up leaning on
the direct input path, and accuracy tracks how separable the classes already
were.

Two and three classes work. Ten do not.

This is the honest result of the architecture, not a tuning failure. It's also
the same lesson the reference experiments kept producing: *it can only learn
what it can already tell apart.*

---

## API

### `Config`

Immutable dataclass. Everything structural lives here; nothing else mutates it.

```python
Config(
    n_inputs=100,          # input vector size
    n_hidden=2000,         # recurrent neurons — this is the capacity knob
    n_classes=10,
    group_size=20,         # output neurons per class (population vote)
    k_in=24, k_rec=20,     # fan-in
    p_inhibitory=0.20,
    target_hz=5.0,         # homeostatic set point
    lr_output=0.02,
)
```

Derived: `.n_outputs`, `.n_neurons`, `.n_synapses`.

### `SpikingNetwork`

```python
net = SpikingNetwork(config, seed=42)

net.step(inputs=None)          # advance 1 ms; returns spike count
net.warmup(steps=4000)         # run with no input; returns rate in Hz
net.predict(x, steps=100)      # -> class index
net.train_step(x, label)       # -> (prediction, correct)
net.learn(reward)              # apply three-factor update
net.reset()                    # clear membrane state, keep weights

net.rate_hz                    # average firing rate
net.n_synapses                 # stored recurrent synapses
net.homeostatic_gain           # current multiplicative gain
net.dopamine_baseline          # running reward estimate
net.summary()

net.save("model.npz")
SpikingNetwork.load("model.npz", seed=42)
```

### `Task`

Subclass and implement `sample`:

```python
from idlebrain import Task

class MyTask(Task):
    n_inputs = 100
    n_classes = 4
    noise = 0.1

    def sample(self, label, rng):
        return make_example(label, rng)
```

Bundled: `PatternTask` (ten 10×10 glyphs), `ArrayTask` (your own templates).

`Task.evaluate(net, n_per_class=20)` returns accuracy; `Task.confusion(...)`
returns the confusion matrix.

### `Trainer`

```python
trainer = Trainer(net, task, steps_per_episode=100, seed=0)
history = trainer.fit(epochs=3000, eval_every=500, n_eval=12)
trainer.evaluate(n_per_class=40)
```

---

## Command line

```bash
idlebrain --hidden 2000 --classes 3 --epochs 3000 --save model.npz
idlebrain --eval-only model.npz

python -m idlebrain --classes 2 --epochs 800
```

---

## Design notes

**Why population voting.** Ten output neurons for ten classes means one winner
per timestep and near-total winner-take-all. Giving each class `group_size`
neurons and summing their spike counts lifted ten-class accuracy from 10% to
35%.

**Why two plastic pathways.** The hidden layer is a random reservoir — it
carries almost no class information (0.9% separability). The direct
`input → output` projection carries the rest. Both learn with the same rule;
the input path just has more signal to work with.

**Why an excitation term and a gain term.** The gain multiplies synaptic
current, the excitation term shifts resting potential. A neuron with weak bias
can never reach threshold on gain alone, no matter how large the gain grows —
it needs the additive term.

---

## Testing

```bash
python -m unittest discover -s tests -v
```

37 tests, ~10 s on a laptop. Covers dynamics (does it really start from
silence), learning (does 2-class training reach 100%), persistence, and
argument validation.

---

## Limitations

- **Ten classes is not solved.** 35%, and it stalls there.
- **The hidden layer doesn't learn.** No STDP, no unsupervised pretraining.
  Adding it (WTA + adaptive thresholds, Diehl & Cook 2015 style) is the
  obvious next step and would likely push ten classes past 80%.
- **Single-threaded, CPU only.** ~0.1 ms/step at 2000 hidden neurons.
- **Fixed 1 ms timestep.** No adaptive integration.

---

## License

MIT. See [LICENSE](LICENSE).
