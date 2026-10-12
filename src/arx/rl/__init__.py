"""Reinforcement-learning demonstration environments.

Imports here are numpy-only. The Gymnasium environment is built lazily inside
:func:`make_goodhart_gym_env`, so ``import arx.rl`` never requires the ``[rl]``
extra.
"""

from .goodhart_env import (
    BanditOutcome,
    GoodhartTracker,
    make_goodhart_gym_env,
    simulate_bandit,
    unrankable_pairs,
)

__all__ = [
    "BanditOutcome",
    "GoodhartTracker",
    "make_goodhart_gym_env",
    "simulate_bandit",
    "unrankable_pairs",
]
