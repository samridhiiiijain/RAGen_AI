from genai_assignment.config import Settings
from genai_assignment.p2.pipeline import generate_outputs, judge_suite


def test_fixture_pipeline_creates_report() -> None:
    output_dir = "assignment/outputs/test_p2_report"
    settings = Settings(ASSIGNMENT_OUTPUT_DIR=output_dir)

    outputs = generate_outputs(use_fixtures=True, settings=settings)
    report = judge_suite(use_fixtures=True, settings=settings)

    assert len(outputs) == 20
    assert report.total_cases == 10
    assert report.wins["grounded"] >= report.wins["baseline"]
    assert report.declared_winner in {"baseline", "grounded", "tie", "no reliable winner"}
    assert (settings.output_dir / "p2_generated_outputs.jsonl").exists()
    assert (settings.output_dir / "p2_pairwise_verdicts.jsonl").exists()
    assert (settings.output_dir / "p2_suite_report.json").exists()
