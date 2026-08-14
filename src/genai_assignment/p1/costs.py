import csv
from dataclasses import dataclass
from pathlib import Path

from genai_assignment.config import get_settings


@dataclass(frozen=True)
class CostScenario:
    vectors: int
    qdrant_storage_gb: float
    qdrant_monthly_usd: float
    pinecone_monthly_usd: float
    notes: str


def estimate_costs(output_path: Path | None = None) -> list[CostScenario]:
    settings = get_settings()
    scenarios = [
        _estimate(vectors=100_000),
        _estimate(vectors=1_000_000),
        _estimate(vectors=10_000_000),
    ]
    output_path = output_path or settings.output_dir / "p1_cost_estimates.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CostScenario.__dataclass_fields__.keys()))
        writer.writeheader()
        for scenario in scenarios:
            writer.writerow(scenario.__dict__)
    return scenarios


def _estimate(vectors: int) -> CostScenario:
    # all-MiniLM-L6-v2 has 384 float32 values. Add 60 percent payload/index overhead.
    raw_gb = vectors * 384 * 4 / (1024**3)
    storage_gb = raw_gb * 1.6

    # Self-hosted Qdrant rough monthly VM assumptions for discussion.
    if vectors <= 100_000:
        qdrant_monthly = 24.0
        qdrant_note = "small 2 vCPU VM plus disk"
    elif vectors <= 1_000_000:
        qdrant_monthly = 80.0
        qdrant_note = "4 vCPU VM plus SSD disk"
    else:
        qdrant_monthly = 320.0
        qdrant_note = "larger memory VM or small cluster"

    # Pinecone serverless-style rough comparator for assignment discussion.
    pinecone_monthly = round(max(25.0, storage_gb * 0.33 + vectors / 1_000_000 * 15.0), 2)

    return CostScenario(
        vectors=vectors,
        qdrant_storage_gb=round(storage_gb, 3),
        qdrant_monthly_usd=round(qdrant_monthly, 2),
        pinecone_monthly_usd=pinecone_monthly,
        notes=qdrant_note,
    )
