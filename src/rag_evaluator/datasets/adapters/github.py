from __future__ import annotations

from pathlib import Path

from rag_evaluator.datasets.adapters.base import DatasetAdapter
from rag_evaluator.datasets.normalizers import build_normalizer
from rag_evaluator.io import load_jsonl
from rag_evaluator.schemas import EvalSample


class GitHubDatasetAdapter(DatasetAdapter):
    """
    Adapter for GitHub-hosted datasets after they are downloaded locally.
    """

    def load(self) -> list[EvalSample]:
        """
        Load local JSONL records and normalize them using the configured normalizer.
        """

        dataset_path = self._dataset_path()
        normalizer_key = str(self.config.metadata.get("normalizer", self.config.name))
        normalizer = build_normalizer(normalizer_key, self.config)
        rows = load_jsonl(dataset_path)
        samples: list[EvalSample] = []

        for index, record in enumerate(rows):
            samples.append(
                normalizer.normalize_record(
                    record,
                    index=index,
                    split=self.config.split,
                )
            )

            if self.config.sample_limit is not None and len(samples) >= self.config.sample_limit:
                break

        return samples
    
    def _dataset_path(self) -> str:
        if self.config.path is not None:
            return self.config.path

        fallback = self.config.metadata.get("local_normalized_path")
        if isinstance(fallback, str) and fallback.strip():
            return str(Path(fallback))

        raise ValueError(
            "GitHub-backed datasets require config.path or metadata['local_normalized_path']."
        )
