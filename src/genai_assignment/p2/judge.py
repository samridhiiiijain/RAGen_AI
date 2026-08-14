import json
import re
import time
from typing import Any

from pydantic import ValidationError

from genai_assignment.p2.models import GeneratedOutput, JudgeCase, PairwiseVerdict, RubricScore


RUBRIC = """You are a strict factual evaluator. Your job is not to reward fluent writing.
Before choosing a winner, compare each answer against the Facts and Expected output.

Hard rules:
- Unsupported claims must be penalized even if the answer sounds helpful.
- A shorter fully grounded answer beats a longer answer with invented details.
- Do not prefer Answer A or Answer B because of position.
- If AB and BA order could reasonably change your preference, use "tie".
- If both answers are correct and equivalent, use "tie".

Score each answer from 1 to 5 on:
- correctness: factual accuracy against expected output and facts
- faithfulness: whether claims are supported by provided facts
- completeness: covers the important parts of the request
- instruction_following: follows system prompt and criteria
- tone_safety: professional, concise, and safe

Pairwise winner rules:
- Prefer the answer with higher correctness and faithfulness.
- If one answer has unsupported claims and the other does not, prefer the grounded answer.
- If both are materially equivalent, return "tie".
- Penalize unsupported confident claims even if they sound polished."""


class QwenJudge:
    def __init__(self, api_key: str, model: str) -> None:
        from groq import Groq

        self.model = model
        self.client = Groq(api_key=api_key)

    def judge_pair(
        self,
        case: JudgeCase,
        first: GeneratedOutput,
        second: GeneratedOutput,
        order: str,
    ) -> PairwiseVerdict:
        prompt = build_pairwise_prompt(case, first.output, second.output)
        completion = self._create_with_retry(prompt)
        raw = completion.choices[0].message.content or ""
        usage_obj = getattr(completion, "usage", None)
        verdict = parse_pairwise_verdict(
            raw,
            case_id=case.id,
            order=order,
            config_a=first.config_name,
            config_b=second.config_name,
            judge_prompt=prompt,
        )
        verdict.usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
            "completion_tokens": getattr(usage_obj, "completion_tokens", None),
            "total_tokens": getattr(usage_obj, "total_tokens", None),
        }
        return verdict

    def _create_with_retry(self, prompt: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a strict evaluator. Return only JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                last_error = exc
                if "rate_limit" not in str(exc).lower() and "429" not in str(exc):
                    raise
                time.sleep(min(20, 2 + attempt * 4))
        raise RuntimeError("Groq judge call failed after rate-limit retries.") from last_error


class GeminiPairwiseJudge:
    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self.model = model
        self.client = genai.Client(api_key=api_key)

    def judge_pair(
        self,
        case: JudgeCase,
        first: GeneratedOutput,
        second: GeneratedOutput,
        order: str,
    ) -> PairwiseVerdict:
        prompt = build_pairwise_prompt(case, first.output, second.output)
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"You are a strict evaluator. Return only JSON.\n\n{prompt}",
            config={"temperature": 0.0, "response_mime_type": "application/json"},
        )
        raw = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        verdict = parse_pairwise_verdict(
            raw,
            case_id=case.id,
            order=order,
            config_a=first.config_name,
            config_b=second.config_name,
            judge_prompt=prompt,
        )
        verdict.usage = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "completion_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
        }
        return verdict


def build_pairwise_prompt(case: JudgeCase, answer_a: str, answer_b: str) -> str:
    expected = case.expected_output or "No reference answer provided."
    criteria = case.criteria or "Use the general rubric."
    return f"""{RUBRIC}

Input:
{case.input}

System prompt given to generator:
{case.system_prompt}

Facts:
{case.facts}

Expected output:
{expected}

Task-specific criteria:
{criteria}

Answer A:
{answer_a}

Answer B:
{answer_b}

Return exactly this JSON object:
{{
  "winner": "a" | "b" | "tie",
  "scores_a": {{
    "correctness": 1-5,
    "faithfulness": 1-5,
    "completeness": 1-5,
    "instruction_following": 1-5,
    "tone_safety": 1-5
  }},
  "scores_b": {{
    "correctness": 1-5,
    "faithfulness": 1-5,
    "completeness": 1-5,
    "instruction_following": 1-5,
    "tone_safety": 1-5
  }},
  "rationale": "short reason",
  "evidence_a": ["supported claim or quote from A"],
  "evidence_b": ["supported claim or quote from B"],
  "unsupported_a": ["unsupported or contradicted claim from A, or empty list"],
  "unsupported_b": ["unsupported or contradicted claim from B, or empty list"],
  "confidence": 0.0-1.0
}}"""


def parse_pairwise_verdict(
    raw: str,
    case_id: str,
    order: str,
    config_a: str,
    config_b: str,
    judge_prompt: str = "",
) -> PairwiseVerdict:
    parse_status = "ok"
    try:
        payload = _loads_json(raw)
    except json.JSONDecodeError:
        payload = _fallback_payload()
        parse_status = "invalid_json_fallback"

    try:
        winner = payload.get("winner", "tie")
        if winner not in {"a", "b", "tie"}:
            winner = "tie"
            parse_status = "invalid_winner_fallback"
        preferred_config = None
        if winner == "a":
            preferred_config = config_a
        elif winner == "b":
            preferred_config = config_b
        return PairwiseVerdict(
            case_id=case_id,
            order=order,  # type: ignore[arg-type]
            winner=winner,
            preferred_config=preferred_config,
            scores_a=RubricScore.model_validate(payload.get("scores_a", _neutral_scores())),
            scores_b=RubricScore.model_validate(payload.get("scores_b", _neutral_scores())),
            rationale=str(payload.get("rationale", "No valid rationale returned."))[:1000],
            evidence_a=[str(item)[:300] for item in payload.get("evidence_a", [])],
            evidence_b=[str(item)[:300] for item in payload.get("evidence_b", [])],
            unsupported_a=[str(item)[:300] for item in payload.get("unsupported_a", [])],
            unsupported_b=[str(item)[:300] for item in payload.get("unsupported_b", [])],
            confidence=float(payload.get("confidence", 0.0)),
            judge_prompt=judge_prompt,
            raw_text=raw,
            parse_status=parse_status,
        )
    except (TypeError, ValueError, ValidationError):
        return PairwiseVerdict(
            case_id=case_id,
            order=order,  # type: ignore[arg-type]
            winner="tie",
            preferred_config=None,
            scores_a=RubricScore(**_neutral_scores()),
            scores_b=RubricScore(**_neutral_scores()),
            rationale="Malformed judge output; treated as tie.",
            evidence_a=[],
            evidence_b=[],
            unsupported_a=[],
            unsupported_b=[],
            confidence=0.0,
            judge_prompt=judge_prompt,
            raw_text=raw,
            parse_status="schema_fallback",
        )


def fixture_judge_pair(
    case: JudgeCase,
    first: GeneratedOutput,
    second: GeneratedOutput,
    order: str,
) -> PairwiseVerdict:
    score_first = _fixture_score(case, first.output)
    score_second = _fixture_score(case, second.output)
    winner = "tie"
    if score_first > score_second:
        winner = "a"
    elif score_second > score_first:
        winner = "b"
    preferred = first.config_name if winner == "a" else second.config_name if winner == "b" else None
    return PairwiseVerdict(
        case_id=case.id,
        order=order,  # type: ignore[arg-type]
        winner=winner,
        preferred_config=preferred,
        scores_a=_score_to_rubric(score_first),
        scores_b=_score_to_rubric(score_second),
        rationale="Fixture judge compared overlap with reference and penalized unsupported-looking claims.",
        evidence_a=[],
        evidence_b=[],
        unsupported_a=[],
        unsupported_b=[],
        confidence=0.8 if winner != "tie" else 0.55,
        judge_prompt=build_pairwise_prompt(case, first.output, second.output),
        raw_text='{"fixture": true}',
        parse_status="fixture",
        usage={"total_tokens": 0},
    )


def _loads_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _fallback_payload() -> dict[str, Any]:
    return {
        "winner": "tie",
        "scores_a": _neutral_scores(),
        "scores_b": _neutral_scores(),
        "rationale": "Judge returned malformed JSON; verdict treated as tie.",
        "evidence_a": [],
        "evidence_b": [],
        "unsupported_a": [],
        "unsupported_b": [],
        "confidence": 0.0,
    }


def _neutral_scores() -> dict[str, int]:
    return {
        "correctness": 3,
        "faithfulness": 3,
        "completeness": 3,
        "instruction_following": 3,
        "tone_safety": 3,
    }


def _fixture_score(case: JudgeCase, output: str) -> int:
    expected_terms = set((case.expected_output or "").lower().replace(".", "").split())
    output_terms = set(output.lower().replace(".", "").split())
    overlap = len(expected_terms.intersection(output_terms))
    unsupported_flags = ["30 days", "24 hours", "automatically enabled", "whenever they want", "including urgent"]
    penalty = 2 if any(flag in output.lower() for flag in unsupported_flags) else 0
    score = min(5, max(1, 2 + overlap // 3 - penalty))
    return score


def _score_to_rubric(score: int) -> RubricScore:
    return RubricScore(
        correctness=score,
        faithfulness=score,
        completeness=max(1, min(5, score)),
        instruction_following=max(1, min(5, score)),
        tone_safety=5 if score >= 3 else 4,
    )
