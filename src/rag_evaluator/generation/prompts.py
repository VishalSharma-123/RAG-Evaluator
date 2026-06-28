from __future__ import annotations

from rag_evaluator.schemas import Chunk, EvalSample

SYSTEM_PROMPT = """You are a RAG answer generator.

 Use only the provided context chunks.
 If the context is insufficient, say you do not know.
 Do not invent facts.
 Return a concise answer grounded in the retrieved evidence.
 """

def build_context_block(chunks: list[Chunk]) -> str:
    """
    Render retrieved chunks into a deterministic prompt block.
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

def build_generation_prompt(sample: EvalSample, chunks: list[Chunk]) -> str:
    """
    Build the user prompt for grounded answer generation.
    :param sample:
    :param chunks:
    :return:
    """
    context_block = build_context_block(chunks)
    
    return "\n\n".join(
        [
            f"Question: {sample.question}",
            f"Question type: {sample.question_type}",
            f"Answerable: {sample.is_answerable}",
            "Instructions:",
            "- Answer using only the provided context.",
            "- If the answer cannot be supported by the context, reply with `I don't know`.",
            "- Keep the answer concise and directly responsive to the question.",
            "Context:",
            context_block or "[No context retrieved]",
            "Answer:",
        ]
    )