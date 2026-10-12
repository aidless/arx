"""A minimal Goodhart demonstration.

The mechanism, in one paragraph. There are ``n_arms`` actions with fixed
*true* qualities. The agent never sees true quality; it sees ``true + noise``,
because the score comes from a judge rather than from the task. Any policy that
optimises the observed score is therefore partly optimising the judge's noise.
Once the policy's advantage collapses to within the noise standard deviation,
further optimisation cannot distinguish "better arm" from "lucky draw" — and
gradient steps from that point on are fitting residuals. ``sigma`` *is* the
noise floor, so arms differing by less than ``sigma`` are unrankable in
principle, not merely in this sample.

This is the same quantity the paper's noise-floor audit estimates from votes;
here it is constructed rather than measured, which is what makes it useful as a
control: you know the answer, so you can check whether a detector finds it.

Two entry points:

* :func:`simulate_bandit` — pure numpy, no RL library. Runs anywhere, and is
  what the tests use.
* :func:`make_goodhart_gym_env` — a Gymnasium environment for real PPO runs.
  Requires the ``[rl]`` extra; raises a clear error when it is missing rather
  than silently degrading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GoodhartTracker:
    """Accumulates (observed, true) pairs and reports the Goodhart signature."""

    observed: list[float] = field(default_factory=list)
    true: list[float] = field(default_factory=list)

    def record(self, observed: float, true: float) -> None:
        self.observed.append(float(observed))
        self.true.append(float(true))

    def noise_floor(self, window: int = 50) -> float:
        """Std of the observed signal over the trailing window.

        Once the policy has plateaued, this *is* the floor: the spread the
        agent is still able to exploit.
        """
        if len(self.observed) < window:
            return float("nan")
        return float(np.std(self.observed[-window:]))

    def divergence(self, window: int = 50) -> float:
        """Correlation between observed and true reward over the window.

        Near zero or negative while the observed reward is still rising is the
        signature: the policy is moving along a direction the true objective
        does not share.
        """
        if len(self.observed) < window:
            return float("nan")
        o = np.asarray(self.observed[-window:], dtype=float)
        t = np.asarray(self.true[-window:], dtype=float)
        if o.std() == 0.0 or t.std() == 0.0:
            return float("nan")
        return float(np.corrcoef(o, t)[0, 1])

    def is_goodharting(self, window: int = 50, corr_threshold: float = 0.2) -> bool:
        d = self.divergence(window)
        return (not math.isnan(d)) and d < corr_threshold

    def to_dict(self) -> dict:
        return {
            "n": len(self.observed),
            "noise_floor": self.noise_floor(),
            "divergence": self.divergence(),
            "is_goodharting": self.is_goodharting(),
        }


@dataclass
class BanditOutcome:
    """Result of a simulated bandit run."""

    true_quality: list[float]
    sigma: float
    #: Fraction of the final window spent on arms within one sigma of the best.
    time_on_indistinguishable_arms: float
    best_arm: int
    chosen_arm: int
    regret: float
    noise_floor: float
    divergence: float
    is_goodharting: bool
    n_rounds: int

    @property
    def ranked_correctly(self) -> bool:
        return self.chosen_arm == self.best_arm

    def to_dict(self) -> dict:
        return {
            "true_quality": [round(q, 4) for q in self.true_quality],
            "sigma": self.sigma,
            "best_arm": self.best_arm,
            "chosen_arm": self.chosen_arm,
            "ranked_correctly": self.ranked_correctly,
            "regret": round(self.regret, 4),
            "noise_floor": round(self.noise_floor, 4),
            "divergence": round(self.divergence, 4),
            "is_goodharting": self.is_goodharting,
            "time_on_indistinguishable_arms": round(
                self.time_on_indistinguishable_arms, 4
            ),
            "n_rounds": self.n_rounds,
        }


def simulate_bandit(
    *,
    n_arms: int = 10,
    n_rounds: int = 400,
    sigma: float = 0.05,
    epsilon: float = 0.1,
    seed: int = 0,
    window: int = 100,
) -> BanditOutcome:
    """epsilon-greedy on the *observed* reward. Pure numpy, no RL library.

    The point of the demo is what happens at the end: the policy keeps pulling
    an arm whose true quality is within ``sigma`` of the best, and the observed
    reward keeps ticking up, while true quality does not. The detector
    (``GoodhartTracker``) is the thing under test, so the ground truth is
    available on purpose.
    """
    if n_arms < 2:
        raise ValueError("n_arms must be >= 2")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")

    rng = np.random.default_rng(seed)
    true_quality = rng.uniform(0.3, 1.0, size=n_arms)
    best_arm = int(np.argmax(true_quality))

    counts = np.zeros(n_arms)
    sums = np.zeros(n_arms)
    tracker = GoodhartTracker()
    last_choices: list[int] = []

    for _ in range(n_rounds):
        if rng.random() < epsilon or counts.min() == 0:
            arm = int(rng.integers(n_arms))
        else:
            means = np.where(counts > 0, sums / np.maximum(counts, 1), -np.inf)
            arm = int(np.argmax(means))
        observed = float(true_quality[arm] + rng.normal(0.0, sigma))
        counts[arm] += 1
        sums[arm] += observed
        tracker.record(observed, float(true_quality[arm]))
        last_choices.append(arm)

    means = sums / np.maximum(counts, 1)
    chosen_arm = int(np.argmax(means))

    tail = last_choices[-window:]
    tolerance = sigma
    indistinguishable = int(
        np.sum(np.abs(true_quality[tail] - true_quality[best_arm]) <= tolerance)
    )
    regret = float(true_quality[best_arm] - true_quality[chosen_arm])

    return BanditOutcome(
        true_quality=[float(q) for q in true_quality],
        sigma=float(sigma),
        time_on_indistinguishable_arms=indistinguishable / max(len(tail), 1),
        best_arm=best_arm,
        chosen_arm=chosen_arm,
        regret=regret,
        noise_floor=tracker.noise_floor(window=min(window, max(2, n_rounds // 2))),
        divergence=tracker.divergence(window=min(window, max(2, n_rounds // 2))),
        is_goodharting=tracker.is_goodharting(
            window=min(window, max(2, n_rounds // 2))
        ),
        n_rounds=n_rounds,
    )


def unrankable_pairs(true_quality, sigma: float) -> list[tuple[int, int]]:
    """Arm pairs whose true qualities differ by less than the noise floor.

    These are the pairs no amount of data from this judge can order. Listing
    them is the whole argument of the audit, in miniature.
    """
    q = np.asarray(true_quality, dtype=float)
    out: list[tuple[int, int]] = []
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            if abs(q[i] - q[j]) < sigma:
                out.append((i, j))
    return out


def make_goodhart_gym_env(*, n_arms: int = 10, sigma: float = 0.05, seed: int = 0):
    """A Gymnasium environment for real RL runs. Requires the ``[rl]`` extra."""
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "make_goodhart_gym_env needs the rl extra: "
            "pip install 'arx-researcher[rl]'"
        ) from exc

    rng = np.random.default_rng(seed)
    true_q = rng.uniform(0.3, 1.0, size=n_arms)
    best = int(np.argmax(true_q))

    class GoodhartEnv(gym.Env):  # type: ignore[misc]
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.action_space = spaces.Discrete(n_arms)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(n_arms,), dtype=np.float32
            )
            self._steps = 0
            self.tracker = GoodhartTracker()

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._steps = 0
            return true_q.astype(np.float32), {}

        def step(self, action):
            arm = int(action)
            true_r = float(true_q[arm])
            observed = true_r + float(rng.normal(0.0, sigma))
            self.tracker.record(observed, true_r)
            self._steps += 1
            terminated = False
            truncated = self._steps >= 200
            info = {
                "true_reward": true_r,
                "noise_floor": sigma,
                "regret": float(true_q[best] - true_r),
            }
            return true_q.astype(np.float32), observed, terminated, truncated, info

    return GoodhartEnv()


__all__ = [
    "BanditOutcome",
    "GoodhartTracker",
    "make_goodhart_gym_env",
    "simulate_bandit",
    "unrankable_pairs",
]
