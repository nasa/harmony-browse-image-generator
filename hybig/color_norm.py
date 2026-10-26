"""Vendored color normalization utilities.

Minimal replacements for matplotlib.colors.Normalize and
matplotlib.colors.BoundaryNorm, covering only the behavior
used in browse.py.
"""

import numpy as np


class Normalize:
    """Linear normalization from [vmin, vmax] to [0, 1].

    NaN values in the input pass through as NaN.
    """

    def __init__(self, vmin=None, vmax=None):
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, value):
        result = np.asarray(value, dtype=float)
        if self.vmin == self.vmax:
            return np.zeros_like(result)
        with np.errstate(invalid='ignore'):
            return np.clip((result - self.vmin) / (self.vmax - self.vmin), 0.0, 1.0)


class BoundaryNorm:
    """Map values to integer indices based on a sorted sequence of boundaries.

    A value in [boundaries[i], boundaries[i+1]) maps to index i.
    Values below boundaries[0] clip to 0; values at or above boundaries[-1]
    clip to ncolors - 1.
    """

    def __init__(self, boundaries, ncolors):
        self.boundaries = np.asarray(boundaries, dtype=float)
        self.ncolors = ncolors

    def __call__(self, value):
        xx = np.asarray(value, dtype=float)
        return np.digitize(xx, self.boundaries) - 1
