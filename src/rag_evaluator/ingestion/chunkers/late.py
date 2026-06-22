from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_evaluator.ingestion.chunkers.base import Chunker, SourceDocument
from rag_evaluator.ingestion.chunkers.utils import chunk_id, split_sentences_with_offsets
from rag_evaluator.schemas import Chunk


@dataclass(frozen=True)
class LateChunker(Chunker):
    """
    Late-style chunker that expands each anchor window with nearby sentence context.
    """

    chunk_size: int = 512
    context_sentences: int = 1

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        if self.context_sentences < 0:
            raise ValueError("context_sentences must be >= 0")

    def chunk(self, documents: Sequence[SourceDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for document in documents:
            sentences = split_sentences_with_offsets(document.text)
            if not sentences:
                continue

            anchor_start = 0
            chunk_index = 0

            while anchor_start < len(sentences):
                anchor_end = anchor_start
                anchor_char_count = 0

                while anchor_end < len(sentences):
                    sentence = sentences[anchor_end][0]
                    next_total = anchor_char_count + len(sentence)
                    if anchor_end > anchor_start and next_total > self.chunk_size:
                        break

                    anchor_char_count = next_total
                    anchor_end += 1

                left = max(0, anchor_start - self.context_sentences)
                right = min(len(sentences), anchor_end + self.context_sentences)

                start_char = sentences[left][1]
                end_char = sentences[right - 1][2]
                anchor_start_char = sentences[anchor_start][1]
                anchor_end_char = sentences[anchor_end - 1][2]
                chunk_text = document.text[start_char:end_char]

                if chunk_text.strip():
                    metadata = dict(document.metadata or {})
                    metadata.update(
                        {
                            "chunking_strategy": "late",
                            "anchor_start_char": anchor_start_char,
                            "anchor_end_char": anchor_end_char,
                        }
                    )
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

                anchor_start = anchor_end

        return chunks
