from __future__ import annotations

from rag_evaluator.schemas import Chunk, EvalSample, GeneratedAnswer

SYSTEM_PROMPT = """You are a strict RAG evaluation judge.

You must score the generated answer against the question, the expected answer when available, and the provided context chunks.

Return JSON only. Do not include markdown fences, commentary, or extra keys.
Use this exact object shape:
{
  "faithfulness": 0.0,
  "relevance": 0.0,
  "hallucination": 0.0,
  "bert_score": 0.0
}

Scoring rules:
- All scores must be numeric values between 0 and 1 inclusive.
- faithfulness: how well the answer is supported by the provided context and/or reference answer.
- relevance: how directly the answer addresses the question.
- hallucination: how much unsupported or invented content appears in the answer. 0 means none, 1 means severe.
- bert_score: semantic similarity to the reference answer when a reference answer is available; otherwise return null.

Be conservative:
- Do not reward unsupported claims.
- If the answer is empty, give low faithfulness/relevance and high hallucination.
- If the question is unanswerable and the answer correctly abstains, score faithfulness high and hallucination low.
- If the answer is partially correct, reflect that with intermediate scores.
"""

def build_context_block(chunks: list[Chunk]) -> str:
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

def build_judge_prompt(
        sample: EvalSample,
        generated_answer: GeneratedAnswer,
        context_chunks: list[Chunk],
) -> str:
    context_block = build_context_block(context_chunks)
    reference_answer = sample.reference_answer if sample.reference_answer else "[No reference answer]"
    answer_text = generated_answer.answer.strip() or "[Empty answer]"

    return "\n\n".join(
        [
            f"Question: {sample.question}",
            f"Question type: {sample.question_type.value}",
            f"Answerable: {sample.is_answerable}",
            f"Reference answer: {reference_answer}",
            f"Generated answer: {answer_text}",
            "Context:",
            context_block or "[No context provided]",
            "Return only JSON with keys: faithfulness, relevance, hallucination, bert_score.",
        ]
    )
