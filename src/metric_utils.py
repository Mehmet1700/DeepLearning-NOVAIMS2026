"""Helpers for sklearn-compatible metric inputs."""

import numpy as np


def prepare_probabilities_for_sklearn(probabilities):
    """Return class probabilities as numeric float32 arrays for sklearn metrics."""
    return np.asarray(probabilities, dtype=np.float32)
