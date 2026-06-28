from __future__ import annotations

from rag_evaluator.schemas import Chunk, QuestionType

SYSTEM_PROMPT = """You create synthetic evaluation samples for a RAG evaluation framework.

Your job is to generate high-quality question-answer examples grounded only in the
provided source chunks.

Rules:
- Use only the provided chunk content.
- Do not invent facts not present in the chunks.
- Every answerable sample must cite valid evidence_chunk_ids from the provided chunks.
- Unanswerable samples must have question_type `unanswerable`.
- Return strict JSON only.
"""


def build_chunk_block(chunks: list[Chunk]) -> str:
    """
    Render source chunks intro a deterministic prompt block.
    :param chunks:
    :return:
    """
    sections: list[str] = []
    
    for index, chunk in enumerate(chunks, start=1):
        sections.append(
            "\n".join(
                [
                    f"[Chunk {index}]",
                    f"chunk_id: {chunk.chunk_id}",
                    f"document_id: {chunk.document_id}",
                    "text:",
                    chunk.text,
                ]
            )
        )
    return "\n\n".join(sections)


def build_synthetic_generation_prompt(
        chunks: list[Chunk],
        *,
        question_types: list[QuestionType] | None = None,
        max_samples: int | None = None
) -> str:
    """
    Build the user prompt for synthetic EvalSample generation.
    :param chunks:
    :param question_types:
    :param max_samples:
    :return:
    """
    allowed_types = question_types or list(QuestionType)
    type_values = ", ".join(question_type.value for question_type in allowed_types)
    sample_count = str(max_samples) if max_samples is not None else 'as many high-quality samples as appropriate'  # noqa: E501
    
    return "\n\n".join(
        [
            f"Allowed question types: {type_values}",
            f"Target sample count: {sample_count}",
            "Generate synthetic evaluation samples grounded in the chunks below.",
            "Return a JSON object with this shape:",
            (
                "{\n"
                '  "samples": [\n'
                "    {\n"
                '      "sample_id": "string",\n'
                '      "question": "string",\n'
                '      "reference_answer": "string or null",\n'
                '      "question_type": "factoid | multi_hop | abstractive | adversarial | comparative | unanswerable",\n'  # noqa: E501
                '      "evidence_chunk_ids": ["chunk_id"],\n'
                '      "is_answerable": true,\n'
                '      "metadata": {}\n'
                "    }\n"
                "  ]\n"
                "}"
            ),
            "Field requirements:",
            "- sample_id must be unique within this response.",
            "- question must be realistic, specific, and useful for RAG evaluation.",
            "- question_type must be one of the allowed question types only.",
            "- Every answerable sample must include a non-empty reference_answer.",
            "- Every answerable sample must include at least one evidence_chunk_id.",
            "- Every evidence_chunk_id must exactly match one of the provided chunk IDs.",
            "- For unanswerable samples: set is_answerable to false, set question_type to unanswerable, set reference_answer to null, and use an empty evidence_chunk_ids list.",  # noqa: E501
            "- metadata may be an empty object if there is nothing useful to add.",
            "Quality requirements:",
            "- Prefer questions that require the model to use the provided evidence, not general world knowledge.",  # noqa: E501
            "- Do not generate duplicate or near-duplicate questions.",
            "- Keep answers concise and grounded in the cited evidence.",
            "- Use multi_hop only when answering requires combining information from multiple chunks.",  # noqa: E501
            "- Use comparative only when the question compares two or more entities or facts from the chunks.",  # noqa: E501
            "- Use adversarial only when the question is intentionally tricky but still answerable from the chunks.",  # noqa: E501
            "Chunks:",
            build_chunk_block(chunks) or "[No chunks provided]",
        ]
    )
