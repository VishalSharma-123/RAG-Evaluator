from __future__ import annotations

from pathlib import Path

from rag_evaluator.ingestion.chunkers import SourceDocument, build_chunker
from rag_evaluator.io import write_jsonl
from rag_evaluator.schemas import Chunk


def build_chunks(
    *,
    chunker_type: str,
    chunk_size: int | None,
    chunk_overlap: int,
    text: str | None,
    input_path: Path | None,
    document_id: str,
) -> list[Chunk]:
    if text is None and input_path is None:
        raise ValueError("Provide either --text or --input.")

    if text is not None and input_path is not None:
        raise ValueError("Use only one of --text or --input.")

    content = text
    if input_path is not None:
        content = input_path.read_text(encoding="utf-8")

    assert content is not None
    chunker = build_chunker(
        chunker_type=chunker_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    document = SourceDocument(
        document_id=document_id,
        text=content,
        metadata={"source_path": str(input_path) if input_path is not None else None},
    )
    return chunker.chunk([document])


def write_chunks(output_path: Path, chunks: list[Chunk]) -> int:
    write_jsonl(output_path, chunks)
    print(f"Wrote {len(chunks)} chunks to {output_path}")
    return 0
