"""Parity checks for the accelerated bootstrap against the repository reference.

Run from the repository root:
    python -m unittest discover -s Filipowicz_beads -p 'test_analysis_helpers.py'
"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utilities as reference
from filipowicz_analysis_helpers import bootstrap_ci


class BootstrapParity(unittest.TestCase):
    def test_reference_percentiles_including_degenerate_responses(self):
        rng = np.random.default_rng(12)
        cases = [
            (rng.integers(2, size=493), rng.integers(2, size=493)),
            (np.zeros(493, dtype=int), rng.integers(2, size=493)),
            (np.ones(493, dtype=int), np.ones(493, dtype=int)),
            (np.r_[np.zeros(492, dtype=int), 1], rng.integers(2, size=493)),
        ]
        for response, target in cases:
            with self.subTest(response_counts=np.bincount(response).tolist()):
                expected = reference.get_Iry_CI(response, target, (2.5, 97.5),
                                                100, seed=234, N_threads=1)
                actual = bootstrap_ci(response, target, (2.5, 97.5), 100, seed=234)
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
