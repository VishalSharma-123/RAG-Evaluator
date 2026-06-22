from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_evaluator.ingestion.chunkers.base import Chunker, SourceDocument
from rag_evaluator.ingestion.chunkers.utils import (
    chunk_id,
    sentence_similarity,
    split_sentences_with_offsets,
)
from rag_evaluator.schemas import Chunk


@dataclass(frozen=True)
class SemanticChunker(Chunker):
    """
    Heuristic semantic chunker that prefers boundaries at topic shifts.
    """

    chunk_size: int = 512
    chunk_overlap_sentences: int = 1
    similarity_threshold: float = 0.15

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        if self.chunk_overlap_sentences < 0:
            raise ValueError("chunk_overlap_sentences must be >= 0")

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

    def chunk(self, documents: Sequence[SourceDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for document in documents:
            sentences = split_sentences_with_offsets(document.text)
            if not sentences:
                continue

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
                        previous_sentence = selected[-1][0]
                        similarity = sentence_similarity(previous_sentence, sentence)
                        if total_chars >= max(1, self.chunk_size // 2):
                            if similarity <= self.similarity_threshold:
                                break
                        else:
                            break

                    selected.append((sentence, start, end))
                    total_chars = next_total
                    cursor += 1

                    if cursor < len(sentences) and total_chars >= self.chunk_size:
                        similarity = sentence_similarity(
                            selected[-1][0],
                            sentences[cursor][0],
                        )
                        if similarity <= self.similarity_threshold:
                            break

                if not selected:
                    break

                start_char = selected[0][1]
                end_char = selected[-1][2]
                chunk_text = document.text[start_char:end_char]

                if chunk_text.strip():
                    metadata = dict(document.metadata or {})
                    metadata["chunking_strategy"] = "semantic"
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id(document.document_id, chunk_index),
                            document_id=document.document_id,
                            text=chunk_text,
                            start_char=start_char,
                            end_char=end_char,
                            metadata=metadata,
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
