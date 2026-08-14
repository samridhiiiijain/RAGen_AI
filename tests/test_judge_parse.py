from genai_assignment.p2.judge import parse_pairwise_verdict


def test_parse_json_inside_fence() -> None:
    raw = """```json
{"winner":"b","scores_a":{"correctness":2,"faithfulness":2,"completeness":2,"instruction_following":3,"tone_safety":5},"scores_b":{"correctness":5,"faithfulness":5,"completeness":4,"instruction_following":5,"tone_safety":5},"rationale":"B is grounded.","confidence":0.9}
```"""

    verdict = parse_pairwise_verdict(raw, case_id="x", order="ab", config_a="baseline", config_b="grounded")

    assert verdict.winner == "b"
    assert verdict.preferred_config == "grounded"
    assert verdict.parse_status == "ok"


def test_malformed_json_becomes_tie() -> None:
    verdict = parse_pairwise_verdict("not json", case_id="x", order="ab", config_a="baseline", config_b="grounded")

    assert verdict.winner == "tie"
    assert verdict.preferred_config is None
    assert verdict.parse_status == "invalid_json_fallback"
