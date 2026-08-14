import json
import time
from pathlib import Path
from typing import Any


def now_ms() -> int:
    return int(time.time() * 1000)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


class Timer:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 3)
