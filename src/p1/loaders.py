import hashlib
import re
from pathlib import Path

from genai_assignment.p1.models import SourceDocument


SUPPORTED_SUFFIXES = {".md", ".markdown", ".html", ".htm", ".pdf"}


def load_documents(corpus_dir: Path) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = _load_text(path)
        metadata = _extract_metadata(text)
        text_without_meta = _strip_metadata_line(text)
        checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
        docs.append(
            SourceDocument(
                source_id=path.stem,
                path=path,
                text=_normalize_whitespace(text_without_meta),
                metadata=metadata,
                checksum=checksum,
            )
        )
    return docs


def _load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return path.read_text(encoding="utf-8")
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
        return soup.get_text(separator="\n")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported source type: {path}")


def _extract_metadata(text: str) -> dict[str, str]:
    match = re.search(r"Metadata:\s*(.+)", text, flags=re.IGNORECASE)
    if not match:
        return {}
    metadata: dict[str, str] = {}
    for part in match.group(1).split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _strip_metadata_line(text: str) -> str:
    return re.sub(r"^\s*Metadata:\s*.+$", "", text, flags=re.IGNORECASE | re.MULTILINE)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
