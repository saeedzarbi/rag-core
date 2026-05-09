from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    i = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(text=chunk, chunk_index=i))
            i += 1
        start += step

    return chunks
