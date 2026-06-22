from __future__ import annotations

from typing import Any

from rag_evaluator.datasets.adapters.base import DatasetAdapter
from rag_evaluator.datasets.normalizers import build_normalizer
from rag_evaluator.schemas import EvalSample


class HuggingFaceDatasetAdapter(DatasetAdapter):
    """
    Adapter for Hugging Face datasets that need normalization to EvalSample.
    """

    def load(self) -> list[EvalSample]:
        """
        Load a Hugging Face dataset split and normalize records.
        """

        if self.config.dataset_name is None:
            raise ValueError("Hugging Face datasets require config.dataset_name.")

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Hugging Face dataset loading requires the `datasets` package. "
                "Install it with: python -m pip install datasets"
            ) from exc

        metadata = self.config.metadata
        dataset_path = str(metadata.get("loader_path") or self.config.dataset_name)
        dataset_kwargs: dict[str, Any] = {}

        if self.config.dataset_config is not None and dataset_path not in {"json", "parquet"}:
            dataset_kwargs["name"] = self.config.dataset_config

        loader_kwargs = metadata.get("loader_kwargs")
        if isinstance(loader_kwargs, dict):
            dataset_kwargs.update(loader_kwargs)

        data_files = metadata.get("data_files")
        if data_files is not None:
            dataset_kwargs["data_files"] = data_files

        sample_limit = self.config.sample_limit
        use_streaming = sample_limit is not None

        raw_dataset = load_dataset(
            dataset_path,
            split=self.config.split,
            streaming=use_streaming,
            **dataset_kwargs,
        )
        normalizer_key = str(self.config.metadata.get("normalizer", self.config.name))
        normalizer = build_normalizer(normalizer_key, self.config)

        samples: list[EvalSample] = []
        for index, record in enumerate(raw_dataset):
            samples.append(
                normalizer.normalize_record(
                    dict(record),
                    index=index,
                    split=self.config.split,
                )
            )

            if self.config.sample_limit is not None and len(samples) >= self.config.sample_limit:
                break

        return samples
