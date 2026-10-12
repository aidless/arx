"""Noise-floor estimators.

The noise floor is the accuracy a gold generator achieves against the gold
it helped produce. If the top contestant does not beat that number, the
leaderboard's ordering is measuring the ruler, not the systems.

Three estimators:

``loo_floor``            leave-one-generator-out agreement with the consensus
``external_gold_floor``  agreement with a gold set disjoint from the generators
``dawid_skene``          latent-truth EM, for when there is no external gold

All of them are pure functions of a vote matrix. No LLM, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .verdict import UndecidableReason


def _observed(votes: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_less(votes, 0)


def column_majority(votes: np.ndarray, tie_break: int | None = None) -> np.ndarray:
    """Per-item majority label over non-negative votes.

    Ties are broken deterministically (smallest label by default). Determinism
    matters: the whole point of the gate is that a third party re-running it
    gets the same verdict, so a tie must never be resolved by dict ordering or
    an RNG.

    Items with no votes get ``-1``.
    """
    votes = np.asarray(votes)
    if votes.ndim != 2:
        raise ValueError("votes must be 2-D (generators x items)")
    m, n = votes.shape
    out = np.full(n, -1, dtype=int)
    for j in range(n):
        col = votes[:, j]
        col = col[col >= 0]
        if col.size == 0:
            continue
        labels, counts = np.unique(col, return_counts=True)
        top = counts.max()
        winners = labels[counts == top]
        if tie_break is not None and tie_break in winners:
            out[j] = int(tie_break)
        else:
            out[j] = int(winners.min())
    return out


def loo_floor(votes: np.ndarray) -> np.ndarray:
    """Leave-one-generator-out floor, one number per generator.

    For generator ``g`` we rebuild the gold from the *other* generators and
    measure how often ``g`` agrees with it. This is the honest floor: the
    generator never gets to grade its own contribution.

    Returns an array of length ``m``. Entries are ``nan`` where the remaining
    generators produced no consensus (which is the ``m < 2`` degeneracy).

    NOTE on m=2: removing one generator leaves exactly one, whose "consensus"
    is its own answer. The value then equals the pairwise agreement rate and no
    longer means self-reproduction against a consensus. Callers must treat
    ``m == 2`` as :data:`UndecidableReason.FLOOR_DEGENERATE`; this function
    still returns the number so the degeneracy is visible, not silent.
    """
    votes = np.asarray(votes)
    if votes.ndim != 2:
        raise ValueError("votes must be 2-D (generators x items)")
    m, n = votes.shape
    floors = np.full(m, np.nan, dtype=float)
    for g in range(m):
        others = np.delete(votes, g, axis=0)
        if others.size == 0:
            continue
        consensus = column_majority(others)
        row = votes[g]
        mask = (row >= 0) & (consensus >= 0)
        if not mask.any():
            continue
        floors[g] = float((row[mask] == consensus[mask]).mean())
    return floors


def external_gold_floor(votes: np.ndarray, external_gold: np.ndarray) -> np.ndarray:
    """Floor measured against a gold set the generators did not produce.

    This is the strongest floor when it is available: there is no self-grading
    at all, so no degeneracy at any ``m``.
    """
    votes = np.asarray(votes)
    gold = np.asarray(external_gold)
    if gold.shape[0] != votes.shape[1]:
        raise ValueError("external_gold must have one label per item")
    mask = gold >= 0
    if not mask.any():
        return np.full(votes.shape[0], np.nan)
    return np.array(
        [float((votes[g][mask] == gold[mask]).mean()) for g in range(votes.shape[0])]
    )


@dataclass
class DawidSkeneResult:
    """Latent-truth EM fit.

    ``complement_flipped`` records that the fit started anti-correlated with
    the majority vote and was corrected. Dawid-Skene likelihood is invariant
    under relabelling every class, so an unanchored EM can converge to the
    complement of the truth and silently invert every downstream accuracy.
    """

    truth: np.ndarray
    theta: np.ndarray  # (m, K, K): P(vote=v | truth=t)
    pi: np.ndarray  # (K,)
    labels: np.ndarray  # class label for each latent index
    n_iter: int
    converged: bool
    log_likelihood: float
    complement_flipped: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def accuracies(self) -> np.ndarray:
        """Mean P(vote = truth) per generator, averaged over latent classes."""
        K = len(self.labels)
        return np.array(
            [float(np.mean([self.theta[g, t, t] for t in range(K)]))
             for g in range(self.theta.shape[0])]
        )

    def usable(self) -> bool:
        if not self.converged:
            return False
        acc = self.accuracies
        # All-uninformative fit: every generator at chance.
        k = len(self.labels)
        if np.all(acc <= 1.0 / k + 1e-6):
            return False
        # Perfectly identical generators: no information to separate them.
        if self.theta.shape[0] > 1 and np.allclose(self.theta, self.theta[0]):
            return False
        return True

    def to_text(self) -> str:
        lines = [
            f"DawidSkene(judges={self.theta.shape[0]}, classes={len(self.labels)}, "
            f"iters={self.n_iter}, converged={self.converged}, "
            f"ll={self.log_likelihood:.2f}, complement_flipped={self.complement_flipped})"
        ]
        for g, acc in enumerate(self.accuracies):
            errs = 1.0 - acc
            lines.append(f"  judge {g}: p_correct={acc:.3f} (err={errs:.3f})")
        if self.warnings:
            lines.append("  warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)


def dawid_skene(
    votes: np.ndarray,
    *,
    max_iters: int = 200,
    tol: float = 1e-8,
    smoothing: float = 1e-3,
    anchor: bool = True,
) -> DawidSkeneResult:
    """Latent-truth Dawid-Skene EM with majority anchoring.

    ``votes``: ``(m, n)`` integer matrix, ``-1`` = abstention.

    Anchoring: initialise from the column majority and, after convergence,
    check the fit against that majority. If the latent labels are mostly
    anti-aligned, flip the class permutation. Without this, EM on sparse or
    balanced data converges to the complement and every accuracy below is
    inverted.
    """
    votes = np.asarray(votes)
    if votes.ndim != 2:
        raise ValueError("votes must be 2-D (generators x items)")
    m, n = votes.shape
    if m == 0 or n == 0:
        raise ValueError("empty vote matrix")

    labels = np.unique(votes[votes >= 0])
    if labels.size == 0:
        raise ValueError("no observed votes")
    K = int(labels.size)
    index = {int(lab): i for i, lab in enumerate(labels)}

    V = np.full((m, n), -1, dtype=int)
    for g in range(m):
        for j in range(n):
            v = int(votes[g, j])
            if v >= 0:
                V[g, j] = index[v]

    # ---- initialisation -------------------------------------------------
    pi = np.full(K, 1.0 / K)
    theta = np.tile(0.75 * np.eye(K) + 0.25 / K, (m, 1, 1))

    if anchor:
        maj = column_majority(votes)
        obs = maj >= 0
        if obs.any():
            post = np.full((n, K), 1.0 / K)
            post[obs] = 0.0
            for j in np.where(obs)[0]:
                post[j, index[int(maj[j])]] = 1.0
        else:
            post = np.full((n, K), 1.0 / K)
    else:
        post = np.full((n, K), 1.0 / K)

    prev_ll = -np.inf
    converged = False
    # Kept explicit rather than read off the loop variable afterwards: the
    # count that matters is "how many M/E passes actually ran", and a future
    # edit that adds an early exit must not silently change it.
    n_iterations = 0

    for iteration in range(1, max_iters + 1):
        n_iterations = iteration
        # ---- M-step ------------------------------------------------------
        pi = (post.sum(axis=0) + smoothing) / (n + smoothing * K)
        for g in range(m):
            for t in range(K):
                w = post[:, t]
                for v in range(K):
                    sel = V[g] == v
                    num = w[sel].sum() + smoothing
                    den = w.sum() + smoothing * K
                    theta[g, t, v] = num / den
        theta = np.clip(theta, 1e-12, 1.0)
        theta /= theta.sum(axis=2, keepdims=True)

        # ---- E-step ------------------------------------------------------
        log_post = np.tile(np.log(pi)[None, :], (n, 1))
        for g in range(m):
            row = V[g]
            for j in range(n):
                v = row[j]
                if v >= 0:
                    log_post[j] += np.log(theta[g, :, v])
        log_post -= log_post.max(axis=1, keepdims=True)
        post = np.exp(log_post)
        post /= post.sum(axis=1, keepdims=True)

        ll = float(np.log(np.maximum(post, 1e-300)).sum() + log_post.max())
        if abs(ll - prev_ll) < tol:
            converged = True
            prev_ll = ll
            break
        prev_ll = ll

    # ---- complement check ----------------------------------------------
    flipped = False
    warnings: list[str] = []
    hard = post.argmax(axis=1)
    maj = column_majority(votes)
    obs = maj >= 0
    if anchor and obs.any():
        maj_idx = np.array([index[int(lab)] for lab in maj[obs]])
        agree = float((hard[obs] == maj_idx).mean())
        if agree < 0.5 and K == 2:
            # Binary case: swapping class roles is the whole degeneracy.
            post = post[:, ::-1]
            theta = theta[:, ::-1, ::-1]
            pi = pi[::-1]
            hard = post.argmax(axis=1)
            flipped = True
            warnings.append(
                "EM converged to the complement of the majority vote; "
                "classes were swapped to restore orientation"
            )
        elif agree < 0.5:
            warnings.append(
                f"latent labels agree with majority only {agree:.2f} of the time; "
                "orientation is ambiguous for K>2"
            )

    truth = labels[hard]
    if not converged:
        warnings.append(f"EM hit max_iters={max_iters} without reaching tol={tol}")

    return DawidSkeneResult(
        truth=truth,
        theta=theta,
        pi=pi,
        labels=labels,
        n_iter=n_iterations,
        converged=converged,
        log_likelihood=prev_ll,
        complement_flipped=flipped,
        warnings=warnings,
    )


def floor_estimate(
    votes: np.ndarray,
    *,
    max_iters: int = 200,
    tol: float = 1e-8,
) -> tuple[float, DawidSkeneResult]:
    """Full-fit floor: mean judge accuracy against the latent truth.

    Returns ``(floor, fit)``. Raises ``RuntimeError`` when the fit collapsed,
    because a collapsed fit has no floor to report and returning a number
    anyway is exactly the failure mode this package exists to prevent.

    Raises ``ValueError`` for fewer than three generators. With m <= 2 the
    bias and noise terms are not separately identifiable, so any number the
    EM happens to emit is an artefact of where it stopped — sometimes it
    diverges (caught below) and sometimes it converges to something confident
    and wrong. The degeneracy has to be structural, not incidental, so the
    refusal is explicit rather than left to the optimiser.
    """
    votes = np.asarray(votes)
    if votes.ndim != 2:
        raise ValueError("votes must be 2-D (generators x items)")
    if votes.shape[0] < 3:
        raise ValueError(
            f"a noise-floor estimate needs at least 3 generators, got "
            f"{votes.shape[0]}: with fewer, rater bias and item difficulty are "
            f"not separately identifiable"
        )
    fit = dawid_skene(votes, max_iters=max_iters, tol=tol)
    if not fit.usable():
        raise RuntimeError(
            "Dawid-Skene fit is not usable (collapsed or uninformative): "
            + "; ".join(fit.warnings)
            if fit.warnings
            else "Dawid-Skene fit is not usable (collapsed or uninformative)"
        )
    accs = fit.accuracies
    valid = accs[~np.isnan(accs)]
    if valid.size == 0:
        raise RuntimeError("no generator accuracy could be estimated")
    return float(valid.mean()), fit


def identifiability_diagnostics(votes: np.ndarray) -> dict:
    """Can this vote matrix support a floor estimate at all?

    A caller should refuse to interpret a floor when the generator set is too
    small to separate bias from noise, when a generator never varies, when the
    generators agree so tightly there is no noise left to floor against, or
    when they agree so loosely that the majority they form is noise.

    The two disagreement extremes fail for opposite reasons and both matter:
    0.99 agreement means the "gold" is one opinion copied; agreement at chance
    means there is no opinion at all.
    """
    votes = np.asarray(votes)
    if votes.ndim != 2:
        raise ValueError("votes must be 2-D")
    m, n = votes.shape
    diags: dict = {"n_generators": m, "n_items": int(n), "undecidable": []}

    if m < 3:
        diags["undecidable"].append(UndecidableReason.FLOOR_DEGENERATE.value)

    constant = []
    for g in range(m):
        observed = votes[g][votes[g] >= 0]
        if observed.size and np.unique(observed).size <= 1:
            constant.append(g)
    diags["constant_generators"] = constant
    if constant:
        diags["undecidable"].append(UndecidableReason.FLOOR_DEGENERATE.value)

    observed_all = votes[votes >= 0]
    if observed_all.size:
        chance = 1.0 / max(int(np.unique(observed_all).size), 1)
    else:
        chance = 1.0
    diags["chance_level"] = chance

    if m >= 2:
        agreements = []
        for a in range(m):
            for b in range(a + 1, m):
                mask = (votes[a] >= 0) & (votes[b] >= 0)
                if mask.any():
                    agreements.append(float((votes[a][mask] == votes[b][mask]).mean()))
        diags["pairwise_agreement_mean"] = (
            float(np.mean(agreements)) if agreements else float("nan")
        )
        diags["pairwise_agreement_min"] = (
            float(np.min(agreements)) if agreements else float("nan")
        )
        if agreements and min(agreements) > 0.99:
            diags["undecidable"].append(UndecidableReason.FLOOR_DEGENERATE.value)
        if agreements and float(np.mean(agreements)) <= chance:
            # Raters landing at chance means the majority they form carries no
            # information, so there is no gold for a score to be measured
            # against — however large the resulting "margin" looks.
            diags["chance_level_agreement"] = True
            diags["undecidable"].append(UndecidableReason.FLOOR_DEGENERATE.value)

    diags["undecidable"] = sorted(set(diags["undecidable"]))
    diags["identifiable"] = not diags["undecidable"]
    return diags
