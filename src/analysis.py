"""
Analysis Utilities
===================
Shared computation functions: summary statistics, human loss.
"""

import numpy as np
from .config import TEST_POSITIONS, HUMAN_ERRORS


def compute_summary_stats(errors_dict):
    """Compute means, SEMs, and asymmetry from per-position error dict.

    Args:
        errors_dict: {position_fraction: np.array of signed errors (%)}

    Returns:
        means: list of mean errors at each position
        sems: list of standard errors of the mean
        asymmetry: |early| - |late| in percentage points
    """
    means = [np.mean(errors_dict[p]) for p in TEST_POSITIONS]
    sems = [np.std(errors_dict[p]) / np.sqrt(len(errors_dict[p])) for p in TEST_POSITIONS]
    early_mag = (abs(means[0]) + abs(means[1])) / 2
    late_mag = (abs(means[2]) + abs(means[3])) / 2
    asymmetry = early_mag - late_mag
    return means, sems, asymmetry


def compute_human_loss(means):
    """Compute L1 distance between model means and human benchmark.

    Args:
        means: list of 4 mean signed errors at [20%, 40%, 60%, 80%]

    Returns:
        float: sum of absolute deviations from human data
    """
    return float(np.sum(np.abs(np.array(HUMAN_ERRORS) - np.array(means))))
