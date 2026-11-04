"""Split Markdown documents into small, self-contained chunks.

Each "## " section becomes one chunk (long sections are split by paragraph),
so an answer can always point to the exact section it comes from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 900


@dataclass(frozen=True)
class Chunk:
    id: str
    file: str
    title: str
    section: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.title} › {self.section}"


def _split_long(text: str) -> list[str]:
    parts, current = [], ""
    for paragraph in text.split("\n\n"):
        if current and len(current) + len(paragraph) > MAX_CHARS:
            parts.append(current.strip())
            current = ""
        current += paragraph + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_markdown(text: str, file: str) -> list[Chunk]:
    title, section, buffer, chunks = Path(file).stem, "Introduction", [], []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        for i, part in enumerate(_split_long(body) if body else []):
            chunks.append(Chunk(f"{Path(file).stem}#{len(chunks)}", file, title, section, part))
        buffer.clear()

    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            section = line[3:].strip()
        else:
            buffer.append(line)
    flush()
    return chunks


def load_folder(folder: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        chunks += chunk_markdown(path.read_text(encoding="utf-8"), path.name)
    if not chunks:
        raise FileNotFoundError(f"no Markdown documents found in {folder}")
    return chunks
