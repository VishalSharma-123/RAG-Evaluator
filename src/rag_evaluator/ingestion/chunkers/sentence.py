from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_evaluator.ingestion.chunkers.base import Chunker, SourceDocument
from rag_evaluator.ingestion.chunkers.utils import chunk_id, split_sentences_with_offsets
from rag_evaluator.schemas import Chunk


@dataclass(frozen=True)
class SentenceChunker(Chunker):
    """
    Sentence-aware chunker that groups sentences up to a target character size.
    """

    chunk_size: int = 512
    chunk_overlap_sentences: int = 1

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        if self.chunk_overlap_sentences < 0:
            raise ValueError("chunk_overlap_sentences must be >= 0")

    def chunk(self, documents: Sequence[SourceDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for document in documents:
            sentences = split_sentences_with_offsets(document.text)
            start_sentence = 0
            chunk_index = 0

            while start_sentence < len(sentences):
                selected: list[tuple[str, int, int]] = []
                total_chars = 0
                cursor = start_sentence

                while cursor < len(sentences):
                    sentence, start, end = sentences[cursor]
                    next_total = total_chars + len(sentence)

                    if selected and next_total > self.chunk_size:
                        break

                    selected.append((sentence, start, end))
                    total_chars = next_total
                    cursor += 1

                if not selected:
                    break

                start_char = selected[0][1]
                end_char = selected[-1][2]
                chunk_text = document.text[start_char:end_char]

                if chunk_text.strip():
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id(document.document_id, chunk_index),
                            document_id=document.document_id,
                            text=chunk_text,
                            start_char=start_char,
                            end_char=end_char,
                            metadata=document.metadata or {},
                        )
                    )
                    chunk_index += 1

                if cursor >= len(sentences):
                    break

                start_sentence = max(
                    cursor - self.chunk_overlap_sentences,
                    start_sentence + 1,
                )

        return chunks
