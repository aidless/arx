"""Tests for the LLM layer and the hypothesis parser.

Two things are being protected here:

* **No model by default.** A default install has an empty base URL, so
  `build_llm` returns `NullLLM`, which returns the empty string. The empty
  string means "no model was consulted", and downstream code must be able to
  tell that apart from "the model had nothing to say" -- otherwise a silent
  no-op looks like a result.
* **The parser is strict about what it *found*, not about what it wished for.**
  A reply that contains no hypotheses must parse to an empty list, never to a
  plausible-looking default.
"""

from __future__ import annotations

import json

import pytest

from arx.config import AgentConfig
from arx.hypothesis import (
    Hypothesis,
    containment,
    deduplicate,
    from_texts,
    generate,
    jaccard,
    parse_hypotheses,
    token_set,
)
from arx.llm import (
    NullLLM,
    OpenAICompatibleLLM,
    ScriptedLLM,
    build_llm,
    parse_json_object,
)

# ---------------------------------------------------------------------------
# NullLLM — the default, and the reason nothing is fabricated
# ---------------------------------------------------------------------------

def test_the_default_config_has_no_model():
    cfg = AgentConfig()
    assert cfg.llm_base_url == ""
    assert cfg.llm_model == ""
    llm = build_llm(cfg)
    assert isinstance(llm, NullLLM)
    assert llm.available is False


def test_null_llm_returns_the_empty_string_and_says_why():
    """Empty means *not consulted*. A caller that cannot tell the difference
    will treat a missing model as a quiet, agreeable one."""
    llm = NullLLM("no endpoint configured")
    assert llm.available is False
    assert llm.chat("system", "user") == ""
    assert "not consulted" in llm.reason or "no endpoint" in llm.reason


# ---------------------------------------------------------------------------
# ScriptedLLM
# ---------------------------------------------------------------------------

def test_scripted_llm_returns_its_script_and_records_prompts():
    llm = ScriptedLLM(replies={"needle": "the reply"}, default="")
    assert llm.chat("sys", "a prompt containing needle") == "the reply"
    assert len(llm.prompts) == 1
    assert llm.prompts[0] == ("sys", "a prompt containing needle")


def test_scripted_llm_falls_back_to_its_default():
    llm = ScriptedLLM(replies={}, default="the default")
    assert llm.chat("s", "anything") == "the default"
    assert llm.available is True


# ---------------------------------------------------------------------------
# OpenAICompatibleLLM — request shaping, offline
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_transport(monkeypatch, payload: dict) -> dict:
    captured: dict = {}

    def _open(request, timeout=None):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        captured["headers"] = dict(request.headers)
        captured["timeout"] = timeout
        return _FakeResponse(payload)

    monkeypatch.setattr("arx.llm.urllib.request.urlopen", _open)
    return captured


def test_the_request_carries_the_model_and_both_messages(monkeypatch):
    captured = _capture_transport(
        monkeypatch, {"choices": [{"message": {"content": "hello"}}]})
    cfg = AgentConfig(llm_base_url="http://example.invalid/v1",
                      llm_api_key="k", llm_model="m")
    llm = OpenAICompatibleLLM(cfg)
    assert llm.chat("SYS", "USER") == "hello"
    assert captured["body"]["model"] == "m"
    assert [m["role"] for m in captured["body"]["messages"]] == ["system", "user"]
    assert captured["body"]["messages"][0]["content"] == "SYS"
    assert "Authorization" in captured["headers"]
    assert captured["url"].endswith("/chat/completions")


def test_usage_is_reported_before_the_attempt_not_after(monkeypatch):
    """A timed-out request still consumes the gateway's quota.

    The sink therefore fires *before* the request goes out. A sink that only
    saw successes would under-report exactly when it matters, and that is the
    difference between a budget and a wish.
    """
    seen: list[tuple] = []

    def _boom(request, timeout=None):  # noqa: ANN001
        raise TimeoutError("slow")

    monkeypatch.setattr("arx.llm.urllib.request.urlopen", _boom)
    cfg = AgentConfig(llm_base_url="http://example.invalid/v1", llm_model="m")
    llm = OpenAICompatibleLLM(
        cfg, usage_sink=lambda kind, info: seen.append((kind, info)))
    with pytest.raises(TimeoutError):
        llm.chat("s", "u")
    assert [k for k, _ in seen] == ["llm_call"]
    assert seen[0][1]["model"] == "m"
    assert llm.stats.calls == 1
    assert llm.stats.failures == 1


def test_a_malformed_response_raises_rather_than_returning_junk(monkeypatch):
    _capture_transport(monkeypatch, {"unexpected": "shape"})
    cfg = AgentConfig(llm_base_url="http://example.invalid/v1", llm_model="m")
    with pytest.raises(RuntimeError):
        OpenAICompatibleLLM(cfg).chat("s", "u")


def test_statistics_are_kept(monkeypatch):
    _capture_transport(monkeypatch, {"choices": [{"message": {"content": "x"}}]})
    cfg = AgentConfig(llm_base_url="http://example.invalid/v1", llm_model="m")
    llm = OpenAICompatibleLLM(cfg)
    llm.chat("s", "u")
    llm.chat("s", "u")
    assert llm.stats.calls == 2
    json.dumps(llm.stats.to_dict())


# ---------------------------------------------------------------------------
# parse_json_object
# ---------------------------------------------------------------------------

def test_plain_json_object_parses():
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_fenced_json_parses():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_with_surrounding_prose_parses():
    text = 'Here you go:\n{"a": 1, "b": [2, 3]}\nHope that helps!'
    assert parse_json_object(text) == {"a": 1, "b": [2, 3]}


def test_nested_braces_in_strings_do_not_confuse_the_scanner():
    assert parse_json_object('{"a": "}{"}') == {"a": "}{"}


def test_garbage_returns_none_rather_than_raising():
    assert parse_json_object("no json here") is None
    assert parse_json_object("") is None
    assert parse_json_object("{unclosed") is None


def test_a_json_array_is_not_an_object():
    """The contract is an object; quietly accepting a list would let a caller
    index the wrong thing."""
    assert parse_json_object("[1, 2, 3]") is None


# ---------------------------------------------------------------------------
# tokenisation and similarity
# ---------------------------------------------------------------------------

def test_token_set_drops_case_and_stopwords():
    tokens = token_set("The Reranker Improves the Retrieval Score")
    assert "the" not in tokens
    assert "improves" in tokens


def test_jaccard_bounds():
    assert jaccard("reranking improves retrieval",
                   "reranking improves retrieval") == pytest.approx(1.0)
    assert jaccard("reranking improves retrieval",
                   "cosine schedules beat linear") == pytest.approx(0.0)


def test_very_short_tokens_are_ignored():
    """Two- and three-letter tokens are mostly stopwords or glue; counting
    them would make unrelated short hypotheses look similar."""
    assert token_set("a b is of to") == set()
    assert jaccard("a b", "c d") == pytest.approx(0.0)


def test_containment_ignores_extra_words_where_jaccard_collapses():
    """A short hypothesis restated at length is the same idea.

    Jaccard divides by the union, so every added clause shrinks it; the
    overlap coefficient divides by the smaller set, so the extra words on the
    longer side do not count against the match.
    """
    short = "reranker improves retrieval"
    long = short + " on the long-document split under a fixed budget"
    assert containment(long, short) == pytest.approx(1.0)
    assert containment(short, long) == pytest.approx(1.0)  # min() is symmetric
    assert jaccard(short, long) < 0.5


# ---------------------------------------------------------------------------
# parsing generated hypotheses
# ---------------------------------------------------------------------------

def test_hypotheses_parse_from_the_documented_format():
    reply = json.dumps({
        "hypotheses": [
            {"text": "A reranker improves retrieval on long documents",
             "rationale": "Long documents exceed the encoder window",
             "confidence": 0.7,
             "expected_effect_se_ratio": 2.5},
            {"text": "Larger batches reduce variance",
             "rationale": "Fewer optimizer steps per epoch"},
        ]
    })
    hyps = parse_hypotheses(reply)
    assert len(hyps) == 2
    assert hyps[0].text.startswith("A reranker")
    assert hyps[0].confidence == pytest.approx(0.7)
    assert hyps[0].expected_effect_se_ratio == pytest.approx(2.5)
    assert hyps[0].origin == "llm"
    assert hyps[1].confidence == pytest.approx(0.5)  # default when absent
    assert hyps[1].expected_effect_se_ratio is None


def test_an_empty_reply_parses_to_no_hypotheses():
    """Never invent one. A silent default here is how a run files a finding
    nobody proposed."""
    assert parse_hypotheses("") == []
    assert parse_hypotheses("I could not find any hypotheses.") == []


def test_confidence_is_clamped_and_junk_is_defaulted():
    reply = json.dumps({"hypotheses": [
        {"text": "h", "confidence": 5, "expected_effect_se_ratio": "nonsense"},
    ]})
    hyp = parse_hypotheses(reply)[0]
    assert hyp.confidence == pytest.approx(1.0)
    assert hyp.expected_effect_se_ratio is None


def test_a_non_positive_effect_ratio_is_dropped_not_kept_as_zero():
    """A zero or negative effect/SE ratio would make `plan_k_max` think the
    effect is trivially detectable."""
    reply = json.dumps({"hypotheses": [
        {"text": "h", "expected_effect_se_ratio": 0},
        {"text": "g", "expected_effect_se_ratio": -3},
    ]})
    assert [h.expected_effect_se_ratio for h in parse_hypotheses(reply)] == [None, None]


def test_items_without_text_are_skipped():
    reply = json.dumps({"hypotheses": [
        {"rationale": "no text"}, {"text": "   "}, {"text": "kept"},
    ]})
    assert [h.text for h in parse_hypotheses(reply)] == ["kept"]


def test_a_malformed_hypotheses_field_yields_nothing():
    assert parse_hypotheses(json.dumps({"hypotheses": "not a list"})) == []


def test_hypothesis_round_trips_through_a_dict():
    hyp = Hypothesis(text="h", rationale="r", confidence=0.4,
                     expected_effect_se_ratio=1.5)
    assert Hypothesis.from_dict(hyp.to_dict()) == hyp


def test_from_texts_builds_hypotheses_and_labels_them_manual():
    """A hand-written hypothesis must not be reported as model-generated."""
    hyps = from_texts(["a", "b"])
    assert [h.text for h in hyps] == ["a", "b"]
    assert all(h.origin == "manual" for h in hyps)


def test_from_texts_skips_blanks():
    assert [h.text for h in from_texts(["a", "   ", ""])] == ["a"]


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def test_generate_uses_the_topic_and_returns_parsed_hypotheses():
    llm = ScriptedLLM(default=json.dumps({"hypotheses": [
        {"text": "A reranker improves retrieval", "rationale": "window"}]}))
    hyps = generate(llm, "retrieval quality")
    assert len(hyps) == 1
    assert "retrieval quality" in llm.prompts[0][1]
    assert "falsifiable" in llm.prompts[0][0].lower() or "methodologist" in llm.prompts[0][0].lower()


def test_generate_with_no_model_returns_nothing():
    hyps = generate(NullLLM(), "topic")
    assert hyps == []


def test_generate_returns_nothing_when_the_model_is_unhelpful():
    llm = ScriptedLLM(default="I am not able to help with that.")
    assert generate(llm, "topic") == []


# ---------------------------------------------------------------------------
# deduplication
# ---------------------------------------------------------------------------

def test_exact_restatements_are_dropped():
    hyps = from_texts([
        "A reranker improves retrieval on long documents",
        "A reranker improves retrieval on long documents",
        "Increasing batch size reduces gradient variance",
    ])
    result = deduplicate(hyps)
    assert len(result.kept) == 2
    assert len(result.dropped) == 1
    assert len(result.pairs) == 1
    match, dropped_text, similarity = result.pairs[0]
    assert similarity > 0.99
    assert dropped_text == "A reranker improves retrieval on long documents"


def test_near_restatements_are_dropped():
    hyps = from_texts([
        "A reranker improves retrieval on long documents",
        "Reranking improves retrieval for long documents",
    ])
    assert len(deduplicate(hyps, threshold=0.6).kept) == 1


def test_distinct_hypotheses_survive():
    hyps = from_texts([
        "A reranker improves retrieval on long documents",
        "Larger batches reduce gradient variance",
        "Cosine schedules beat linear schedules",
    ])
    assert len(deduplicate(hyps).kept) == 3


def test_known_hypotheses_from_earlier_runs_suppress_repeats():
    """This is the cheap, offline half of the cross-run duplicate check: a
    hypothesis already tried must not be re-tested as if it were new."""
    fresh = from_texts(["Reranking improves retrieval on long documents"])
    known = ["A reranker improves retrieval on long documents"]
    result = deduplicate(fresh, known=known)
    assert result.kept == []
    assert len(result.dropped) == 1
    assert result.dropped[0].duplicates == [known[0]]
    assert result.pairs[0][0] == known[0]


def test_deduplicate_is_deterministic_and_order_preserving():
    hyps = from_texts(["a b c d", "a b c d", "e f g h"])
    first = deduplicate(hyps)
    second = deduplicate(hyps)
    assert [h.text for h in first.kept] == [h.text for h in second.kept]
    assert first.kept[0].text == "a b c d"


def test_a_rewording_with_different_terms_is_not_caught_and_that_is_stated():
    """The honest limitation of this layer.

    Token overlap matches *wording*. A hypothesis restated with a different
    vocabulary ("context-aware retrieval" for "reranker-v2") slips through,
    and it must be visible in the test suite that it does -- otherwise someone
    will read the dedup log as evidence that no repeat occurred. The semantic
    half of this check is the embedding index in `arx.longterm`.
    """
    fresh = from_texts(["Context-aware retrieval reduces long-document errors"])
    known = ["A reranker improves retrieval on long documents"]
    result = deduplicate(fresh, known=known)
    assert len(result.kept) == 1
    assert result.dropped == []
    assert result.pairs == []


def test_dedup_result_is_json_safe():
    result = deduplicate(from_texts(["a b", "a b"]))
    json.dumps(result.to_dict())
