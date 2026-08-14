from typing import Any, Literal

from pydantic import BaseModel, Field


class JudgeCase(BaseModel):
    id: str
    input: str
    system_prompt: str
    facts: str
    expected_output: str | None = None
    criteria: str | None = None


class GeneratorConfig(BaseModel):
    name: str
    system_suffix: str
    temperature: float = 0.0


class GeneratedOutput(BaseModel):
    case_id: str
    config_name: str
    model: str
    output: str
    prompt: str
    raw_response: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)


class RubricScore(BaseModel):
    correctness: int
    faithfulness: int
    completeness: int
    instruction_following: int
    tone_safety: int


class PairwiseVerdict(BaseModel):
    case_id: str
    order: Literal["ab", "ba"]
    winner: Literal["a", "b", "tie"]
    preferred_config: str | None
    scores_a: RubricScore
    scores_b: RubricScore
    rationale: str
    evidence_a: list[str] = Field(default_factory=list)
    evidence_b: list[str] = Field(default_factory=list)
    unsupported_a: list[str] = Field(default_factory=list)
    unsupported_b: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    judge_prompt: str = ""
    raw_text: str = ""
    parse_status: str = "ok"
    usage: dict[str, Any] = Field(default_factory=dict)


class SuiteReport(BaseModel):
    generator_model: str
    judge_model: str
    config_a: str
    config_b: str
    total_cases: int
    wins: dict[str, int]
    ties: int
    raw_wins: dict[str, int]
    unstable_cases: list[str]
    stable_cases: int
    declared_winner: str
    declared_winner_reason: str
    mean_scores: dict[str, float]
    position_flip_rate: float
    verbosity_probe: dict[str, Any]
    sycophancy_probe: dict[str, Any]
    self_enhancement_probe: dict[str, Any]
    score_anchor_probe: dict[str, Any]
    confidence_intervals: dict[str, Any]
    validation: dict[str, Any]
