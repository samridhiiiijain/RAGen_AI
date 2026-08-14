import time

from genai_assignment.p2.models import GeneratedOutput, GeneratorConfig, JudgeCase


class GeminiSuiteGenerator:
    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self.model = model
        self.client = genai.Client(api_key=api_key)

    def generate(self, case: JudgeCase, config: GeneratorConfig) -> GeneratedOutput:
        prompt = build_generator_prompt(case, config)
        response = self._generate_with_retry(prompt, config.temperature)
        usage = getattr(response, "usage_metadata", None)
        return GeneratedOutput(
            case_id=case.id,
            config_name=config.name,
            model=self.model,
            output=(response.text or "").strip(),
            prompt=prompt,
            raw_response={},
            usage={
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "completion_tokens": getattr(usage, "candidates_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
            },
        )

    def _generate_with_retry(self, prompt: str, temperature: float):
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": temperature},
                )
            except Exception as exc:
                last_error = exc
                text = str(exc).lower()
                retryable = ["429", "503", "quota", "resource_exhausted", "unavailable", "high demand"]
                if not any(marker in text for marker in retryable):
                    raise
                time.sleep(min(30, 5 + attempt * 5))
        raise RuntimeError("Gemini generation failed after quota retries.") from last_error


def build_generator_prompt(case: JudgeCase, config: GeneratorConfig) -> str:
    return f"""{case.system_prompt}
{config.system_suffix}

Facts:
{case.facts}

User request:
{case.input}

Answer:"""


def fixture_outputs(cases: list[JudgeCase], configs: dict[str, GeneratorConfig], model: str) -> list[GeneratedOutput]:
    outputs: list[GeneratedOutput] = []
    for case in cases:
        for name, config in configs.items():
            prompt = build_generator_prompt(case, config)
            if name == "baseline":
                text = _baseline_fixture(case)
            else:
                text = case.expected_output or _grounded_fixture(case)
            outputs.append(
                GeneratedOutput(
                    case_id=case.id,
                    config_name=name,
                    model=f"fixture-{model}",
                    output=text,
                    prompt=prompt,
                    raw_response={"fixture": True},
                    usage={"total_tokens": _rough_tokens(prompt) + _rough_tokens(text)},
                )
            )
    return outputs


def _baseline_fixture(case: JudgeCase) -> str:
    weaker = {
        "j001": "Starter customers can contact AcmeOps support for help, including urgent support requests.",
        "j002": "NimbusDesk Scale exports are retained for 366 days.",
        "j003": "Yes, AI reply suggestions are available in new NimbusDesk workspaces.",
        "j005": "Failed webhook deliveries are retried for 24 hours.",
        "j006": "Yes, Enterprise customers can switch to EU storage whenever they want.",
        "j009": "The AcmeOps mobile app keeps data offline for 30 days.",
    }
    return weaker.get(case.id, case.expected_output or _grounded_fixture(case))


def _grounded_fixture(case: JudgeCase) -> str:
    return "I do not know from the provided facts."


def _rough_tokens(text: str) -> int:
    return max(1, len(text.split()))
