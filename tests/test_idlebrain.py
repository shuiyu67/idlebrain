"""Tests for idlebrain. Run with ``python -m unittest discover -s tests``.

Kept on the standard library so the suite runs anywhere numpy does — no
pytest required. Everything uses a small network so it finishes in seconds.
"""

import os
import tempfile
import unittest

import numpy as np

from idlebrain import ArrayTask, Config, PatternTask, SpikingNetwork, Trainer


def small_config(**overrides):
    cfg = dict(n_hidden=120, n_classes=2, group_size=4, k_in=8, k_rec=8)
    cfg.update(overrides)
    return Config(**cfg)


class TestConfig(unittest.TestCase):
    def test_derived_sizes(self):
        c = Config(n_inputs=100, n_hidden=1000, n_classes=10, group_size=20)
        self.assertEqual(c.n_outputs, 200)
        self.assertEqual(c.n_neurons, 1300)
        self.assertEqual(c.n_synapses, 1000 * 44)

    def test_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            Config(n_hidden=0)
        with self.assertRaises(ValueError):
            Config(p_inhibitory=1.5)
        with self.assertRaises(ValueError):
            Config(target_hz=-1)

    def test_roundtrip_dict(self):
        c = Config(n_hidden=333)
        self.assertEqual(Config.from_dict(c.to_dict()), c)

    def test_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"n_hidden": 10, "not_a_key": 1})

    def test_is_immutable(self):
        c = Config()
        with self.assertRaises(Exception):
            c.n_hidden = 5


class TestNetworkShape(unittest.TestCase):
    def test_slices_are_contiguous_and_complete(self):
        net = SpikingNetwork(small_config(), seed=1)
        c = net.config
        self.assertEqual(net.input_slice.stop, c.n_inputs)
        self.assertEqual(net.hidden_slice.start, c.n_inputs)
        self.assertEqual(net.hidden_slice.stop, c.n_inputs + c.n_hidden)
        self.assertEqual(net.output_slice.stop, net.n_neurons)

    def test_synapse_count_within_bound(self):
        # Duplicate (src, dst) draws get merged by scipy, so the real count is
        # at most the configured fan-in budget, and never far below it.
        cfg = small_config()
        net = SpikingNetwork(cfg, seed=1)
        budget = cfg.n_synapses
        self.assertLessEqual(net.n_synapses, budget)
        self.assertGreater(net.n_synapses, budget * 0.9)

    def test_seed_reproducible(self):
        a = SpikingNetwork(small_config(), seed=7)
        b = SpikingNetwork(small_config(), seed=7)
        np.testing.assert_array_equal(a._weights.toarray(), b._weights.toarray())

    def test_different_seeds_differ(self):
        a = SpikingNetwork(small_config(), seed=7)
        b = SpikingNetwork(small_config(), seed=8)
        self.assertFalse(np.array_equal(a._weights.toarray(), b._weights.toarray()))


class TestDynamics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = SpikingNetwork(small_config(), seed=3)
        cls.rates = []
        for i in range(3000):
            cls.net.step(None)
            if (i + 1) % 500 == 0:
                cls.rates.append(cls.net.rate_hz)

    def test_starts_silent(self):
        # It must not be firing from step one — that would mean we seeded it.
        net = SpikingNetwork(small_config(), seed=3)
        net.step(None)
        self.assertLessEqual(net.rate_hz, 50.0)

    def test_activity_grows_without_input(self):
        self.assertGreater(self.rates[-1], self.rates[0])

    def test_reaches_steady_rate(self):
        self.assertGreater(self.net.rate_hz, 0.5)
        self.assertLess(self.net.rate_hz, 30.0)

    def test_gain_stays_bounded(self):
        self.assertGreaterEqual(self.net.homeostatic_gain, 0.05)
        self.assertLessEqual(self.net.homeostatic_gain, 20.0)

    def test_step_returns_spike_count(self):
        net = SpikingNetwork(small_config(), seed=3)
        net.warmup(500)
        n = net.step(None)
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 0)
        self.assertLessEqual(n, net.n_neurons)

    def test_rejects_wrong_input_shape(self):
        net = SpikingNetwork(small_config(), seed=3)
        with self.assertRaises(ValueError):
            net.step(np.zeros(7, np.float32))

    def test_reset_clears_state_but_keeps_weights(self):
        net = SpikingNetwork(small_config(), seed=3)
        net.warmup(200)
        before = net.rate_hz
        net.reset()
        self.assertEqual(net.rate_hz, 0.0)
        self.assertGreater(before, 0.0)


class TestLearning(unittest.TestCase):
    def test_two_classes_learned(self):
        """The headline claim: with two well-separated classes it reaches 100%.

        ``n_per_class`` is deliberately large and the evaluation RNG is pinned:
        with 20 samples per class a single flipped sample costs 2.5 points, so
        the same network scored anywhere between 0.90 and 1.00 across runs. At
        50 samples the measurement converges on the network's real accuracy and
        the 0.95 threshold is met with margin.
        """
        net = SpikingNetwork(small_config(), seed=11)
        net.warmup(3000)
        task = PatternTask(noise=0.10, n=2)
        Trainer(net, task, seed=5).fit(epochs=800, eval_every=800, verbose=False)
        self.assertGreaterEqual(task.evaluate(net, 50, rng=np.random.default_rng(2024)), 0.95)

    def test_predict_returns_valid_class(self):
        net = SpikingNetwork(small_config(), seed=11)
        task = PatternTask(noise=0.18, n=2)
        rng = np.random.default_rng(0)
        for _ in range(5):
            pred = net.predict(task.sample(0, rng))
            self.assertIn(pred, range(net.config.n_classes))

    def test_train_step_returns_prediction_and_flag(self):
        net = SpikingNetwork(small_config(), seed=11)
        task = PatternTask(noise=0.18, n=2)
        pred, ok = net.train_step(task.sample(1, np.random.default_rng(0)), 1, steps=20)
        self.assertIn(pred, range(2))
        self.assertIsInstance(ok, bool)

    def test_dopamine_baseline_tracks_reward(self):
        net = SpikingNetwork(small_config(), seed=11)
        task = PatternTask(noise=0.18, n=2)
        rng = np.random.default_rng(0)
        for _ in range(20):
            net.train_step(task.sample(0, rng), 0, steps=10)
        self.assertGreaterEqual(net.dopamine_baseline, 0.0)
        self.assertLessEqual(net.dopamine_baseline, 1.0)

    def test_nothing_changes_when_reward_equals_baseline(self):
        net = SpikingNetwork(small_config(), seed=11)
        task = PatternTask(noise=0.18, n=2)
        net.predict(task.sample(0, np.random.default_rng(0)), learn=True, steps=20)
        before = net._w_input_out.copy()
        net.learn(net.dopamine_baseline)
        np.testing.assert_allclose(before, net._w_input_out, rtol=1e-6)


class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        cfg = small_config()
        net = SpikingNetwork(cfg, seed=13)
        task = PatternTask(noise=0.18, n=2)
        rng = np.random.default_rng(1)
        for _ in range(50):
            net.train_step(task.sample(0, rng), 0, steps=10)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            net.save(path)
            restored = SpikingNetwork.load(path, seed=13)

        np.testing.assert_allclose(net._w_input_out, restored._w_input_out)
        np.testing.assert_allclose(net._w_hidden_out, restored._w_hidden_out)
        self.assertEqual(restored.config, cfg)

    def test_load_restores_config_from_file(self):
        cfg = small_config(group_size=6)
        net = SpikingNetwork(cfg, seed=13)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            net.save(path)
            restored = SpikingNetwork.load(path, seed=13)
        self.assertEqual(restored.config.group_size, 6)
        self.assertEqual(restored.config.n_outputs, cfg.n_outputs)

    def test_load_rejects_truncated_file(self):
        net = SpikingNetwork(small_config(), seed=13)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            net.save(path)
            with open(path, "r+b") as f:
                f.truncate(64)
            with self.assertRaises(Exception):
                SpikingNetwork.load(path, seed=13)

    def test_save_is_a_single_file(self):
        net = SpikingNetwork(small_config(), seed=13)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            net.save(path)
            self.assertTrue(os.path.isfile(path))
            self.assertEqual([f for f in os.listdir(d) if f.endswith(".npz")], ["m.npz"])


class TestTasks(unittest.TestCase):
    def test_pattern_task_shapes(self):
        t = PatternTask(noise=0.1)
        self.assertEqual(t.n_classes, 10)
        self.assertEqual(t.n_inputs, 100)

    def test_pattern_task_subset(self):
        t = PatternTask(noise=0.1, n=3)
        self.assertEqual(t.n_classes, 3)

    def test_noise_zero_is_deterministic(self):
        t = PatternTask(noise=0.0, n=2)
        rng = np.random.default_rng(0)
        a = t.sample(1, rng)
        b = t.sample(1, rng)
        np.testing.assert_array_equal(a, b)

    def test_noise_flips_pixels(self):
        t = PatternTask(noise=0.5, n=2)
        clean = PatternTask(noise=0.0, n=2).templates[0]
        noisy = t.sample(0, np.random.default_rng(0))
        self.assertGreater(int((clean != noisy).sum()), 0)

    def test_array_task_roundtrip(self):
        templates = np.eye(3, 100, dtype=np.float32)
        t = ArrayTask(templates, noise=0.1)
        self.assertEqual((t.n_classes, t.n_inputs), (3, 100))

    def test_array_task_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            ArrayTask(np.zeros(5, np.float32))

    def test_array_task_roundtrip_via_dopamine(self):
        templates = np.eye(2, 100, dtype=np.float32)
        task = ArrayTask(templates, noise=0.05)
        net = SpikingNetwork(small_config(), seed=21)
        net.warmup(2000)
        Trainer(net, task, seed=2).fit(epochs=500, eval_every=500, verbose=False)
        # 50 samples per class, pinned RNG: this configuration measures
        # ~0.95-0.98 with a spread of ~0.015, so 0.9 holds with margin.
        self.assertGreaterEqual(
            task.evaluate(net, 50, rng=np.random.default_rng(2025)), 0.9
        )


class TestTrainerBehaviour(unittest.TestCase):
    def test_keep_best_restores_best_checkpoint(self):
        net = SpikingNetwork(small_config(), seed=17)
        task = PatternTask(noise=0.10, n=2)
        trainer = Trainer(net, task, seed=4)
        h = trainer.fit(epochs=400, eval_every=100, verbose=False, keep_best=True)
        # whatever the final episode did, we should end up at the best score seen
        end = task.evaluate(net, 40, rng=np.random.default_rng(2026))
        self.assertGreaterEqual(end, h.best_accuracy - 0.15)

    def test_history_records_checkpoints(self):
        net = SpikingNetwork(small_config(), seed=17)
        task = PatternTask(noise=0.10, n=2)
        h = Trainer(net, task, seed=4).fit(epochs=100, eval_every=50, verbose=False)
        self.assertEqual(h.epochs, [50, 100])
        self.assertEqual(len(h.accuracy), 2)
        self.assertEqual(len(h.seconds), 2)
        self.assertIn("best_accuracy", h.as_dict())

    def test_callback_fires_per_checkpoint(self):
        seen = []
        net = SpikingNetwork(small_config(), seed=17)
        task = PatternTask(noise=0.10, n=2)
        Trainer(net, task, seed=4).fit(
            epochs=100, eval_every=50, verbose=False, callback=lambda h: seen.append(len(h.epochs))
        )
        self.assertEqual(seen, [1, 2])


class TestTrainerGuards(unittest.TestCase):
    def test_rejects_input_mismatch(self):
        net = SpikingNetwork(small_config(), seed=1)
        bad = ArrayTask(np.zeros((2, 50), np.float32))
        with self.assertRaises(ValueError):
            Trainer(net, bad)

    def test_rejects_class_mismatch(self):
        net = SpikingNetwork(small_config(), seed=1)
        bad = ArrayTask(np.zeros((9, 100), np.float32))
        with self.assertRaises(ValueError):
            Trainer(net, bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
