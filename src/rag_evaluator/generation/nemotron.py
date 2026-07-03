from __future__ import annotations

from dataclasses import dataclass

from rag_evaluator.generation.chat_completion import ChatCompletionGenerator


@dataclass(frozen=True)
class NemotronGenerator(ChatCompletionGenerator):
    """
    Backward-compatible alias for the generic chat-completions generator.
    """

    pass
