import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from genai_assignment.config import Settings, get_settings
from genai_assignment.p2.generator import GeminiSuiteGenerator, fixture_outputs
from genai_assignment.p2.io import read_cases, read_generator_configs, read_jsonl_dicts, write_jsonl
from genai_assignment.p2.judge import GeminiPairwiseJudge, QwenJudge, fixture_judge_pair
from genai_assignment.p2.models import GeneratedOutput, PairwiseVerdict, SuiteReport


def generate_outputs(use_fixtures: bool = False, settings: Settings | None = None) -> list[GeneratedOutput]:
    settings = settings or get_settings()
    cases = read_cases(Path("assignment/data/p2/suite.jsonl"))
    configs = read_generator_configs(Path("assignment/data/p2/generator_configs.json"))

    if use_fixtures:
        outputs = fixture_outputs(cases, configs, settings.generator_model)
    else:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required unless --use-fixtures is set.")
        generator = GeminiSuiteGenerator(settings.gemini_api_key, settings.generator_model)
        outputs = [
            generator.generate(case, config)
            for case in cases
            for config in configs.values()
        ]

    write_jsonl(settings.output_dir / "p2_generated_outputs.jsonl", outputs)
    return outputs


def judge_suite(
    use_fixtures: bool = False,
    judge_engine: str = "qwen",
    settings: Settings | None = None,
) -> SuiteReport:
    settings = settings or get_settings()
    cases = read_cases(Path("assignment/data/p2/suite.jsonl"))
    outputs_path = settings.output_dir / "p2_generated_outputs.jsonl"
    if not outputs_path.exists():
        generate_outputs(use_fixtures=use_fixtures, settings=settings)
    outputs = _outputs_by_case(_read_outputs(outputs_path))

    if use_fixtures:
        judge = None
        judge_model = f"fixture-{settings.judge_model}"
    elif judge_engine == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for --judge-engine gemini unless --use-fixtures is set.")
        judge = GeminiPairwiseJudge(settings.gemini_api_key, settings.generator_model)
        judge_model = settings.generator_model
    else:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required unless --use-fixtures is set.")
        judge = QwenJudge(settings.groq_api_key, settings.judge_model)
        judge_model = settings.judge_model

    verdicts: list[PairwiseVerdict] = []
    for case in cases:
        baseline = outputs[case.id]["baseline"]
        grounded = outputs[case.id]["grounded"]
        pairs = [
            ("ab", baseline, grounded),
            ("ba", grounded, baseline),
        ]
        for order, first, second in pairs:
            if judge:
                verdict = judge.judge_pair(case, first, second, order)
            else:
                verdict = fixture_judge_pair(case, first, second, order)
            verdicts.append(verdict)

    suffix = "gemini_judge" if judge_engine == "gemini" and not use_fixtures else "qwen_judge"
    verdict_path = settings.output_dir / f"p2_pairwise_verdicts_{suffix}.jsonl"
    write_jsonl(verdict_path, verdicts)
    write_jsonl(settings.output_dir / "p2_pairwise_verdicts.jsonl", verdicts)
    report = build_report(settings, verdicts, outputs, judge_model=judge_model)
    report_path = settings.output_dir / f"p2_suite_report_{suffix}.json"
    report_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    (settings.output_dir / "p2_suite_report.json").write_text(
        json.dumps(report.model_dump(), indent=2),
        encoding="utf-8",
    )
    return report


def build_report(
    settings: Settings,
    verdicts: list[PairwiseVerdict],
    outputs: dict[str, dict[str, GeneratedOutput]],
    judge_model: str | None = None,
) -> SuiteReport:
    case_preferences = _case_preferences(verdicts)
    raw_preferences = _raw_case_preferences(verdicts)
    wins = Counter(pref for pref in case_preferences.values() if pref in {"baseline", "grounded"})
    raw_wins = Counter(pref for pref in raw_preferences.values() if pref in {"baseline", "grounded"})
    ties = sum(1 for pref in case_preferences.values() if pref == "tie")
    unstable_cases = sorted(case_id for case_id, pref in case_preferences.items() if pref == "unstable")
    declared_winner, declared_reason = _declared_winner(wins, ties, unstable_cases, len(case_preferences))
    mean_scores = _mean_scores(verdicts)
    flip_rate = _position_flip_rate(verdicts)
    validation = _validation(case_preferences)
    confidence_intervals = _confidence_intervals(case_preferences, flip_rate)

    report = SuiteReport(
        generator_model=settings.generator_model,
        judge_model=judge_model or settings.judge_model,
        config_a="baseline",
        config_b="grounded",
        total_cases=len(case_preferences),
        wins={"baseline": wins.get("baseline", 0), "grounded": wins.get("grounded", 0)},
        ties=ties,
        raw_wins={"baseline": raw_wins.get("baseline", 0), "grounded": raw_wins.get("grounded", 0)},
        unstable_cases=unstable_cases,
        stable_cases=len(case_preferences) - len(unstable_cases),
        declared_winner=declared_winner,
        declared_winner_reason=declared_reason,
        mean_scores=mean_scores,
        position_flip_rate=flip_rate,
        verbosity_probe=_verbosity_probe(),
        sycophancy_probe=_sycophancy_probe(),
        self_enhancement_probe=_self_enhancement_probe(outputs),
        score_anchor_probe=_score_anchor_probe(),
        confidence_intervals=confidence_intervals,
        validation=validation,
    )
    return report


def _read_outputs(path: Path) -> list[GeneratedOutput]:
    rows = read_jsonl_dicts(path)
    return [GeneratedOutput.model_validate(row) for row in rows]


def _outputs_by_case(outputs: list[GeneratedOutput]) -> dict[str, dict[str, GeneratedOutput]]:
    grouped: dict[str, dict[str, GeneratedOutput]] = defaultdict(dict)
    for output in outputs:
        grouped[output.case_id][output.config_name] = output
    return grouped


def _case_preferences(verdicts: list[PairwiseVerdict]) -> dict[str, str]:
    by_case: dict[str, list[PairwiseVerdict]] = defaultdict(list)
    for verdict in verdicts:
        by_case[verdict.case_id].append(verdict)

    preferences: dict[str, str] = {}
    for case_id, items in by_case.items():
        preferred = [item.preferred_config or "tie" for item in items]
        unique = set(preferred)
        if len(unique) > 1:
            preferences[case_id] = "unstable"
        elif preferred.count("grounded") > preferred.count("baseline"):
            preferences[case_id] = "grounded"
        elif preferred.count("baseline") > preferred.count("grounded"):
            preferences[case_id] = "baseline"
        else:
            preferences[case_id] = "tie"
    return preferences


def _raw_case_preferences(verdicts: list[PairwiseVerdict]) -> dict[str, str]:
    by_case: dict[str, list[PairwiseVerdict]] = defaultdict(list)
    for verdict in verdicts:
        by_case[verdict.case_id].append(verdict)
    preferences: dict[str, str] = {}
    for case_id, items in by_case.items():
        preferred = [item.preferred_config or "tie" for item in items]
        if preferred.count("grounded") > preferred.count("baseline"):
            preferences[case_id] = "grounded"
        elif preferred.count("baseline") > preferred.count("grounded"):
            preferences[case_id] = "baseline"
        else:
            preferences[case_id] = "tie"
    return preferences


def _declared_winner(
    wins: Counter[str],
    ties: int,
    unstable_cases: list[str],
    total_cases: int,
) -> tuple[str, str]:
    if unstable_cases:
        return (
            "no reliable winner",
            f"{len(unstable_cases)} of {total_cases} cases changed preference across AB/BA order, so the pipeline flags the run for human review.",
        )
    if wins.get("grounded", 0) > wins.get("baseline", 0):
        return "grounded", "Grounded has more stable pairwise wins."
    if wins.get("baseline", 0) > wins.get("grounded", 0):
        return "baseline", "Baseline has more stable pairwise wins."
    return "tie", f"Stable wins are tied or all {ties} cases are ties."


def _mean_scores(verdicts: list[PairwiseVerdict]) -> dict[str, float]:
    totals: dict[str, list[int]] = defaultdict(list)
    for verdict in verdicts:
        first_config = _first_config(verdict)
        second_config = "grounded" if first_config == "baseline" else "baseline"
        totals[first_config].append(_weighted_score(verdict.scores_a))
        totals[second_config].append(_weighted_score(verdict.scores_b))
    return {key: round(mean(values), 3) for key, values in totals.items()}


def _first_config(verdict: PairwiseVerdict) -> str:
    return "baseline" if verdict.order == "ab" else "grounded"


def _weighted_score(scores: Any) -> int:
    return (
        scores.correctness * 35
        + scores.faithfulness * 25
        + scores.completeness * 15
        + scores.instruction_following * 15
        + scores.tone_safety * 10
    )


def _position_flip_rate(verdicts: list[PairwiseVerdict]) -> float:
    by_case: dict[str, dict[str, PairwiseVerdict]] = defaultdict(dict)
    for verdict in verdicts:
        by_case[verdict.case_id][verdict.order] = verdict
    flips = 0
    comparable = 0
    for orders in by_case.values():
        if "ab" not in orders or "ba" not in orders:
            continue
        comparable += 1
        first_pref = orders["ab"].preferred_config or "tie"
        second_pref = orders["ba"].preferred_config or "tie"
        if first_pref != second_pref:
            flips += 1
    return round(flips / comparable, 4) if comparable else 0.0


def _validation(case_preferences: dict[str, str]) -> dict[str, Any]:
    validation_path = Path("assignment/data/p2/human_validation.jsonl")
    labels = read_jsonl_dicts(validation_path)
    compared = []
    for label in labels:
        judge_pref = case_preferences.get(label["id"], "tie")
        human_pref = label["preferred_config"]
        compared.append(
            {
                "id": label["id"],
                "judge": judge_pref,
                "human": human_pref,
                "match": judge_pref == human_pref,
                "requires_review": judge_pref == "unstable",
            }
        )
    agreement = mean(item["match"] for item in compared) if compared else 0.0
    return {
        "method": "Manual label comparison against human_validation.jsonl.",
        "items": len(compared),
        "agreement": round(agreement, 4),
        "wilson_95": _wilson(sum(item["match"] for item in compared), len(compared)),
        "rows": compared,
    }


def _verbosity_probe() -> dict[str, Any]:
    short_answer = "NimbusDesk Scale includes SAML SSO."
    padded_answer = (
        "NimbusDesk Scale includes SAML SSO. This is useful for teams that want centralized identity, "
        "better account governance, smoother onboarding, simpler permission management, and more consistent "
        "access control across their support organization."
    )
    return {
        "probe": "Same factual answer with padded verbosity.",
        "expected": "Judge should not prefer the padded answer solely because it is longer.",
        "answer_a_words": len(short_answer.split()),
        "answer_b_words": len(padded_answer.split()),
    }


def _sycophancy_probe() -> dict[str, Any]:
    return {
        "probe": "User confidently claims AcmeOps Starter has phone support, but facts say it does not.",
        "expected": "Judge should prefer the answer that corrects the false premise.",
        "confident_wrong_answer": "Yes, Starter includes phone support because you are correct.",
        "grounded_answer": "No. AcmeOps does not provide phone support on Starter.",
    }


def _self_enhancement_probe(outputs: dict[str, dict[str, GeneratedOutput]]) -> dict[str, Any]:
    model_families = sorted({output.model for by_config in outputs.values() for output in by_config.values()})
    return {
        "primary_mitigation": "Generator is Gemini family while judge is Qwen family via Groq.",
        "same_family_baseline": "Run the same saved p2_generated_outputs.jsonl through a Gemini judge as an optional same-family comparison.",
        "observed_generator_models": model_families,
    }


def _score_anchor_probe() -> dict[str, Any]:
    return {
        "probe": "Anchor examples define what 1, 3, and 5 mean before pairwise judging.",
        "score_1_anchor": "Confident unsupported answer that contradicts the facts.",
        "score_3_anchor": "Partly correct answer that misses an important limit or condition.",
        "score_5_anchor": "Fully correct, grounded, concise answer matching the task criteria.",
        "clustering_mitigation": "Main decision uses pairwise winner plus score anchors instead of relying only on absolute scores.",
    }


def _confidence_intervals(case_preferences: dict[str, str], flip_rate: float) -> dict[str, Any]:
    total = len(case_preferences)
    grounded_wins = sum(1 for value in case_preferences.values() if value == "grounded")
    baseline_wins = sum(1 for value in case_preferences.values() if value == "baseline")
    ties = sum(1 for value in case_preferences.values() if value == "tie")
    unstable = sum(1 for value in case_preferences.values() if value == "unstable")
    return {
        "method": "Wilson 95 percent intervals for observed rates; suitable for small eval suites.",
        "grounded_win_rate": {
            "rate": round(grounded_wins / total, 4) if total else 0.0,
            "wilson_95": _wilson(grounded_wins, total),
        },
        "baseline_win_rate": {
            "rate": round(baseline_wins / total, 4) if total else 0.0,
            "wilson_95": _wilson(baseline_wins, total),
        },
        "tie_rate": {
            "rate": round(ties / total, 4) if total else 0.0,
            "wilson_95": _wilson(ties, total),
        },
        "unstable_rate": {
            "rate": round(unstable / total, 4) if total else 0.0,
            "wilson_95": _wilson(unstable, total),
        },
        "position_flip_rate": {
            "rate": flip_rate,
            "wilson_95": _wilson(round(flip_rate * total), total),
        },
    }


def _wilson(successes: int, total: int) -> dict[str, float]:
    if total == 0:
        return {"low": 0.0, "high": 0.0}
    z = 1.96
    phat = successes / total
    denom = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denom
    radius = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total) / denom
    return {"low": round(center - radius, 4), "high": round(center + radius, 4)}
