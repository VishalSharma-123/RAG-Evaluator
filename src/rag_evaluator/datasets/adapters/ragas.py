from __future__ import annotations

from rag_evaluator.datasets.adapters.github import GitHubDatasetAdapter


class RagasTestsetAdapter(GitHubDatasetAdapter):
    """
    Adapter for saved RAGAS-generated JSONL testsets.
    """
