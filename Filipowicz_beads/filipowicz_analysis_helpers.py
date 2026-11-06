"""Exact binary NSB bootstrap with reusable count-based entropy estimates.

Resampling and seeds match utilities.get_Iry_CI. Workers cache entropy by
unordered counts because NSB entropy is invariant to category labels.
The reference ndd implementation infers observed marginal cardinalities,
while retaining the declared joint cardinality; preserve that behavior for
constant responses and bootstrap samples with missing categories.
"""
from functools import lru_cache
import numpy as np
import ndd


@lru_cache(maxsize=200000)
def _entropy(counts, cardinality):
    return ndd.entropy(np.asarray([c for c in counts if c > 0]), k=cardinality, estimator='nsb')


def binary_mi(counts):
    """NSB I(R;Y) in bits for the four joint binary category counts."""
    table = np.asarray(counts).reshape(2, 2)
    return (_entropy(tuple(sorted(table.sum(axis=0))), np.count_nonzero(table.sum(axis=0)))
            + _entropy(tuple(sorted(table.sum(axis=1))), np.count_nonzero(table.sum(axis=1)))
            - _entropy(tuple(sorted(table.ravel())), 4)) * np.log2(np.e)


def _bootstrap_chunk(args):
    joint, base_seed, start, stop = args
    values = np.empty(stop - start)
    for i, seed in enumerate(range(start, stop)):
        indices = np.random.default_rng(base_seed + seed).choice(len(joint), len(joint))
        values[i] = binary_mi(np.bincount(joint[indices], minlength=4))
    return values


def bootstrap_ci(response, target, percentiles, iterations, seed=234, pool=None,
                 chunks=4):
    """Reference percentile CI, using the identical trial-index resamples."""
    joint = 2 * np.asarray(response, dtype=int) + np.asarray(target, dtype=int)
    edges = np.linspace(0, iterations, chunks + 1, dtype=int)
    args = [(joint, seed, int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
    samples = list(map(_bootstrap_chunk, args)) if pool is None else pool.map(_bootstrap_chunk, args)
    return np.percentile(np.concatenate(samples), percentiles)
