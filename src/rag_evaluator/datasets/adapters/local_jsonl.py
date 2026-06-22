from __future__ import annotations

from rag_evaluator.datasets.adapters.base import DatasetAdapter
from rag_evaluator.io import load_eval_samples_jsonl
from rag_evaluator.schemas import EvalSample


class LocalJSONLDatasetAdapter(DatasetAdapter):
    """
    Adapter for JSONL files that already contain normalized EvalSample records.
    """

    def load(self) -> list[EvalSample]:
        """
        Load normalized EvalSample records from local JSONL.
        """

        if self.config.path is None:
            raise ValueError("Local JSONL datasets require config.path.")

        samples = load_eval_samples_jsonl(self.config.path)

        if self.config.sample_limit is None:
            return samples

        return samples[: self.config.sample_limit]
