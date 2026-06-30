from __future__ import annotations

from dataclasses import dataclass

from rag_evaluator.scoring.judges.llm import LLMJudge


@dataclass(frozen=True)
class NemotronJudge(LLMJudge):
    """
    Judge wrapper for the approved nvidia/nemotron family.

    Today this reuses the deterministic heuristic path. Later, this can be
    replaced with a Nemotron-backed judge prompt without changing callers.
    """

    def __post_init__(self) -> None:
        if not self.config.model.startswith("nvidia/nemotron"):
            raise ValueError("NemotronJudge requires a model in the `nvidia/nemotron` family.")

    def provider_name(self) -> str:
        return "nemotron"
