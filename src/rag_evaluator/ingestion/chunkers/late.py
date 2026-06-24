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
                anchor_end = self._find_anchor_end(sentences, anchor_start)
                
                expanded_start = max(0, anchor_start - self.context_sentences)
                expanded_end = min(len(sentences), anchor_end + self.context_sentences)
                
                anchor_start_char = sentences[anchor_start][1]
                anchor_end_char = sentences[anchor_end - 1][2]
                expanded_start_char = sentences[expanded_start][1]
                expanded_end_char = sentences[expanded_end - 1][1]
                
                chunk_text = document.text[expanded_start_char:expanded_end_char]
                
                if chunk_text.strip():
                    metadata = dict(document.metadata or {})
                    metadata.update(
                        {
                            "chunking_strategy": "late",
                            "anchor_start_sentence": anchor_start,
                            "anchor_end_sentence": anchor_end,
                            "anchor_start_char": anchor_start_char,
                            "anchor_end_char": anchor_end_char,
                            "expanded_start_sentence": expanded_start,
                            "expanded_end_sentence": expanded_end,
                            "expanded_start_char": expanded_start_char,
                            "expanded_end_char": expanded_end_char,
                        }
                    )

                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id(document.document_id, chunk_index),
                            document_id=document.document_id,
                            text=chunk_text,
                            start_char=expanded_start_char,
                            end_char=expanded_end_char,
                            metadata=metadata,
                        )
                    )
                    chunk_index += 1
                anchor_start = anchor_end
        
        return chunks

    def _find_anchor_end(
            self,
            sentences: list[tuple[str, int, int]],
            anchor_start: int,
    ) -> int:
        anchor_end = anchor_start
        anchor_chars = 0
        
        while anchor_end < len(sentences):
            sentence, _, _ = sentences[anchor_end]
            next_chars = anchor_chars + len(sentence)
            
            if anchor_end > anchor_start and next_chars > self.chunk_size:
                break
            
            anchor_chars = next_chars
            anchor_end += 1
        
        return anchor_end