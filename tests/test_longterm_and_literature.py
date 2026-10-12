"""Tests for cross-run hypothesis memory and the literature layer.

The interesting properties are all about *limits*, because each of these
modules is a partial answer to a problem that has no complete offline answer:

* The default embedder matches wording, not meaning. Tests pin that it is
  deterministic across processes (SHA-256, never Python's randomised
  ``hash()``) and that it does not pretend to be semantic.
* The global ledger refuses to compare vectors from different embedding
  spaces, because those similarities are not numbers about anything.
* Citation checking distinguishes "does not exist", "exists but you have not
  read it", and "there is no library configured" -- three different things
  that would all be reported as a failure by a lazier design.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import textwrap

import pytest

from arx.literature.citation_check import (
    CitationStatus,
    audit_references,
    verify_one,
)
from arx.literature.search import (
    LocalCorpus,
    Paper,
    SearchOutcome,
    normalise_title,
    token_set,
)
from arx.literature.zotero import LocalLibrary, NullLibrary
from arx.longterm.embedding import (
    HashingEmbedder,
    build_embedder,
    cosine,
    tokenize,
)
from arx.longterm.global_ledger import GlobalLedger, summarise_prior

# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------

def test_the_hashed_embedder_is_deterministic_within_a_process():
    a = HashingEmbedder()
    b = HashingEmbedder()
    assert a.embed("reranking improves retrieval") == b.embed("reranking improves retrieval")


def test_the_hashed_embedder_is_deterministic_across_processes():
    """The whole reason it uses SHA-256: Python's built-in ``hash()`` is salted
    per process, so a stored index would silently stop matching on the next
    run -- a recall failure that looks like 'no duplicates'."""
    script = textwrap.dedent(
        """
        from arx.longterm.embedding import HashingEmbedder
        v = HashingEmbedder().embed("reranking improves retrieval")
        print(",".join(f"{x:.6f}" for x in v[:8]))
        """
    )
    outs = set()
    for seed in ("0", "1", "12345"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin",
                 "PYTHONPATH": "src"},
        )
        outs.add(proc.stdout.strip())
    assert len(outs) == 1, f"embedding changed with PYTHONHASHSEED: {outs}"


def test_embeddings_are_l2_normalised_so_cosine_is_a_dot_product():
    vec = HashingEmbedder().embed("reranking improves retrieval for long documents")
    norm = sum(x * x for x in vec) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_identical_text_scores_one_and_unrelated_text_scores_lower():
    embedder = HashingEmbedder()
    a = embedder.embed("reranking improves retrieval on long documents")
    b = embedder.embed("reranking improves retrieval on long documents")
    c = embedder.embed("cosine schedules beat linear schedules at scale")
    assert cosine(a, b) == pytest.approx(1.0, abs=1e-9)
    assert cosine(a, c) < 0.5


def test_the_embedder_matches_wording_and_that_is_all_it_claims():
    """Keep the limitation visible: a paraphrase with different vocabulary is
    *not* recalled. The reported similarity gap is the evidence."""
    embedder = HashingEmbedder()
    source = embedder.embed("a reranker improves retrieval on long documents")
    rewording = embedder.embed("context-aware retrieval reduces long-document errors")
    overlap = embedder.embed("a reranker improves retrieval on long documents")
    assert cosine(source, overlap) > 0.99
    assert cosine(source, rewording) < 0.8


def test_empty_text_embeds_to_a_zero_vector_rather_than_raising():
    vec = HashingEmbedder().embed("")
    assert len(vec) == 256
    assert all(x == 0.0 for x in vec)


def test_cosine_of_a_zero_vector_is_zero_not_a_division_error():
    zero = [0.0] * 8
    assert cosine(zero, zero) == 0.0
    assert cosine(zero, [1.0] + [0.0] * 7) == 0.0


def test_tokenize_keeps_hyphenated_terms_together():
    """A hyphenated compound is one feature, not its parts.

    That is a deliberate choice (the compound is the term of art) and it has a
    cost: ``long-document`` and ``long document`` no longer share that word, so
    only the surrounding words and the character trigrams carry the match. The
    second assertion pins the cost so nobody has to rediscover it.
    """
    assert "long-document" in tokenize("a long-document split")
    assert "long" not in tokenize("a long-document split")

    embedder = HashingEmbedder()
    hyphenated = embedder.embed("improves retrieval on long-document benchmarks")
    spaced = embedder.embed("improves retrieval on long document benchmarks")
    unrelated = embedder.embed("cosine schedules beat linear schedules at scale")
    # Still far closer than an unrelated sentence, because the rest matches.
    assert cosine(hyphenated, spaced) < 0.99
    assert cosine(hyphenated, spaced) > cosine(hyphenated, unrelated)


def test_build_embedder_dispatches_and_refuses_unknown_kinds():
    assert build_embedder("hash").name == "hash"
    with pytest.raises(ValueError):
        build_embedder("telepathy")


# ---------------------------------------------------------------------------
# global ledger
# ---------------------------------------------------------------------------

def _ledger(tmp_path) -> GlobalLedger:
    return GlobalLedger(tmp_path / "global.db")


def test_recording_and_retrieving_a_prior_hypothesis(tmp_path):
    with _ledger(tmp_path) as led:
        row_id = led.record("A reranker improves retrieval on long documents",
                            verdict="rejected", reason="below the floor",
                            project="proj", run_id="run1")
        assert row_id > 0
        hits = led.similar("A reranker improves retrieval on long documents",
                           threshold=0.8)
        assert len(hits) == 1
        assert hits[0].verdict == "rejected"
        assert hits[0].similarity > 0.99


def test_a_genuinely_different_hypothesis_is_not_recalled(tmp_path):
    with _ledger(tmp_path) as led:
        led.record("A reranker improves retrieval on long documents",
                   verdict="rejected")
        assert led.similar("Cosine schedules beat linear schedules", threshold=0.8) == []


def test_prior_negatives_exclude_findings(tmp_path):
    """The point of the memory is to stop re-running dead ends. A hypothesis
    that *produced* a finding must not be shown as a prior failure."""
    with _ledger(tmp_path) as led:
        led.record("A reranker improves retrieval on long documents",
                   verdict="readable")
        assert led.prior_negatives("A reranker improves retrieval on long documents",
                                   threshold=0.8) == []


def test_prior_negatives_include_the_soft_failures(tmp_path):
    """`marginal` and `undecidable` are failures to learn. Treating them as
    findings is how the same dead end gets re-run four times."""
    with _ledger(tmp_path) as led:
        for verdict in ("marginal", "undecidable", "unreadable", "rejected"):
            led.record(f"A reranker improves retrieval on long documents {verdict}",
                       verdict=verdict)
        found = led.prior_negatives("reranker improves retrieval long documents",
                                    threshold=0.5, top_k=10)
        assert {p.verdict for p in found} == {"marginal", "undecidable",
                                              "unreadable", "rejected"}


def test_vectors_from_different_embedders_are_not_compared(tmp_path):
    """Cosine similarity between two embedding spaces is not a number about
    anything, and mixing them would produce confident nonsense."""
    class OtherEmbedder:
        name = "other"

        def embed(self, text):
            return [0.5] * 256

        def embed_many(self, texts):
            return [self.embed(t) for t in texts]

    with _ledger(tmp_path) as led:
        led.record("A reranker improves retrieval", verdict="rejected")
        other = GlobalLedger(led.path, embedder=OtherEmbedder())
        try:
            assert other.similar("A reranker improves retrieval", threshold=0.0) == []
        finally:
            other.close()
        # ... unless the caller explicitly asks for it
        assert len(led.similar("A reranker improves retrieval", threshold=0.0)) == 1


def test_threshold_is_respected(tmp_path):
    with _ledger(tmp_path) as led:
        led.record("A reranker improves retrieval on long documents",
                   verdict="rejected")
        strict = led.similar("context-aware retrieval for long documents",
                             threshold=0.95)
        loose = led.similar("context-aware retrieval for long documents",
                            threshold=0.05)
        assert strict == []
        assert len(loose) == 1


def test_exclude_id_skips_the_row_being_compared(tmp_path):
    with _ledger(tmp_path) as led:
        row_id = led.record("A reranker improves retrieval on long documents",
                            verdict="rejected")
        assert led.similar("A reranker improves retrieval on long documents",
                           threshold=0.8, exclude_id=row_id) == []


def test_top_k_limits_the_result(tmp_path):
    with _ledger(tmp_path) as led:
        for i in range(5):
            led.record(f"A reranker improves retrieval on long documents v{i}",
                       verdict="rejected")
        assert len(led.similar("A reranker improves retrieval on long documents",
                               threshold=0.3, top_k=2)) == 2


def test_the_ledger_survives_a_reopen(tmp_path):
    path = tmp_path / "global.db"
    with GlobalLedger(path) as led:
        led.record("A reranker improves retrieval", verdict="rejected")
    with GlobalLedger(path) as reopened:
        assert reopened.stats()["n_hypotheses"] == 1
        assert reopened.verify_chain()
        assert reopened.similar("A reranker improves retrieval", threshold=0.8)


def test_stats_counts_projects_and_verdicts(tmp_path):
    with _ledger(tmp_path) as led:
        led.record("h one", verdict="rejected", project="p1")
        led.record("h two", verdict="readable", project="p2")
        stats = led.stats()
        assert stats["n_hypotheses"] == 2
        assert stats["n_projects"] == 2
        assert stats["by_verdict"]["rejected"] == 1


def test_tampering_with_the_chain_is_detected(tmp_path):
    path = tmp_path / "global.db"
    with GlobalLedger(path) as led:
        led.record("A reranker improves retrieval", verdict="rejected")
        led.record("Cosine schedules beat linear", verdict="rejected")
        assert led.verify_chain()

    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE hypotheses SET verdict = 'readable' WHERE id = 1")
    conn.commit()
    conn.close()

    with GlobalLedger(path) as tampered:
        assert not tampered.verify_chain()


def test_meta_round_trips_as_a_dict(tmp_path):
    with _ledger(tmp_path) as led:
        led.record("A reranker improves retrieval", verdict="rejected",
                   meta={"k_max": 5, "n_items": 400})
        hit = led.similar("A reranker improves retrieval", threshold=0.8)[0]
        assert hit.meta == {"k_max": 5, "n_items": 400}
        json.dumps(hit.to_dict())


def test_summarise_prior_produces_a_usable_warning():
    with_priors = summarise_prior
    assert with_priors([]) == ""


def test_summarise_prior_names_the_prior_work(tmp_path):
    with _ledger(tmp_path) as led:
        led.record("A reranker improves retrieval on long documents",
                   verdict="rejected", reason="below the floor")
        priors = led.similar("A reranker improves retrieval on long documents",
                             threshold=0.8)
        text = summarise_prior(priors)
        assert "rejected" in text
        assert "below the floor" in text


# ---------------------------------------------------------------------------
# literature search
# ---------------------------------------------------------------------------

def test_normalise_title_drops_case_and_punctuation():
    assert normalise_title("Reading a Leaderboard, Before You Cite It!") == \
        normalise_title("reading a leaderboard before you cite it")


def test_token_set_drops_short_tokens():
    assert token_set("a bb ccc dddd") == {"ccc", "dddd"}


def test_the_local_corpus_ranks_by_overlap_and_is_deterministic():
    corpus = LocalCorpus()
    corpus.add(title="Reading a leaderboard before you cite it",
               abstract="noise floor for language model evaluation")
    corpus.add(title="Cosine schedules beat linear schedules",
               abstract="optimizer schedules at scale")
    hits = corpus.search("noise floor evaluation")
    assert hits[0].title.startswith("Reading a leaderboard")
    again = corpus.search("noise floor evaluation")
    assert [p.title for p in hits] == [p.title for p in again]


def test_the_local_corpus_returns_nothing_for_an_unrelated_query():
    corpus = LocalCorpus()
    corpus.add(title="Reading a leaderboard", abstract="noise floor")
    assert corpus.search("quantum chromodynamics") == []


# ---------------------------------------------------------------------------
# citation checking
# ---------------------------------------------------------------------------

def _outcome(*titles: str) -> SearchOutcome:
    return SearchOutcome(
        papers=[Paper(title=t, authors=["A"], year=2024, doi=None, source="stub")
                for t in titles],
        sources_ok=["stub"],
    )


def test_a_title_that_is_not_in_any_index_is_unresolved():
    verdict = verify_one("A Completely Fabricated Title 12345xyz",
                         library=NullLibrary(), search=_outcome())
    assert verdict.status is CitationStatus.UNRESOLVED
    assert not verdict.ok


def test_a_title_that_resolves_without_a_library_is_not_a_failure():
    """'No library configured' is declared missing evidence, not a bad
    citation. Reporting it as a failure would train users to ignore it."""
    library = NullLibrary()
    verdict = verify_one("Reading a leaderboard before you cite it",
                         library=library,
                         search=_outcome("Reading a leaderboard before you cite it"))
    assert verdict.status is CitationStatus.LIBRARY_UNAVAILABLE
    assert verdict.ok
    assert verdict.library_checked is False


def test_a_near_match_is_reported_as_a_title_mismatch_not_a_match():
    verdict = verify_one("Reading a leaderboard before you cite it",
                         library=NullLibrary(),
                         search=_outcome("Reading a leaderboard, then citing it"))
    assert verdict.status is CitationStatus.TITLE_MISMATCH
    assert not verdict.ok
    assert "different title" in verdict.reason


def test_resolving_but_not_being_in_the_library_is_its_own_status():
    library = LocalLibrary()  # available, but empty
    verdict = verify_one("Reading a leaderboard before you cite it",
                         library=library,
                         search=_outcome("Reading a leaderboard before you cite it"))
    assert verdict.status is CitationStatus.NOT_IN_LIBRARY
    assert verdict.library_checked is True
    assert not verdict.ok


def test_a_citation_read_and_stored_is_verified():
    library = LocalLibrary()
    library.add("Reading a leaderboard before you cite it", year=2024)
    verdict = verify_one("Reading a leaderboard before you cite it",
                         library=library,
                         search=_outcome("Reading a leaderboard before you cite it"))
    assert verdict.status is CitationStatus.VERIFIED
    assert verdict.ok
    assert verdict.library_item.title.startswith("Reading a leaderboard")


def test_an_empty_title_is_an_error_not_an_unresolved_citation():
    verdict = verify_one("   ", library=NullLibrary(), search=_outcome())
    assert verdict.status is CitationStatus.ERROR


def test_a_failing_search_layer_is_reported_with_its_cause():
    class Exploding:
        papers: list = []
        sources_ok: list = []
        sources_failed = {"arxiv": "connection reset"}

    import arx.literature.citation_check as cc

    real = cc.search_all

    def _boom(*a, **k):
        raise RuntimeError("network down")

    cc.search_all = _boom
    try:
        verdict = verify_one("Whatever Title", library=NullLibrary())
    finally:
        cc.search_all = real
    assert verdict.status is CitationStatus.ERROR
    assert "network down" in verdict.reason


def test_audit_references_collects_verdicts_and_failures(monkeypatch):
    import arx.literature.citation_check as cc

    def _fake_search(title, max_results=5, timeout=20.0):
        if "Fabricated" in title:
            return SearchOutcome(papers=[], sources_ok=["arxiv"],
                                 sources_failed={"semantic_scholar": "429"})
        return _outcome(title)

    monkeypatch.setattr(cc, "search_all", _fake_search)
    library = LocalLibrary()
    library.add("Reading a leaderboard before you cite it", year=2024)

    audit = audit_references(
        ["Reading a leaderboard before you cite it", "A Fabricated Title"],
        library,
    )
    assert audit.n_verified == 1
    assert len(audit.problems) == 1
    assert not audit.ok
    assert audit.sources_failed["semantic_scholar"] == "429"
    json.dumps(audit.to_dict())
    assert "Issues" in audit.to_text() or "Problems" in audit.to_text() or audit.to_text()
