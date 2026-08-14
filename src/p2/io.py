import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from genai_assignment.p2.models import GeneratorConfig, JudgeCase

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    rows: list[T] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(model.model_validate_json(line))
    return rows


def write_jsonl(path: Path, rows: list[BaseModel | dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.model_dump() if isinstance(row, BaseModel) else row
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def read_cases(path: Path) -> list[JudgeCase]:
    return read_jsonl(path, JudgeCase)


def read_generator_configs(path: Path) -> dict[str, GeneratorConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: GeneratorConfig.model_validate(value) for key, value in raw.items()}


def read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
