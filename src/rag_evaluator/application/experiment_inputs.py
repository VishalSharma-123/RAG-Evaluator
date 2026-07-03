from __future__ import annotations

from rag_evaluator.application.types import DatasetLoadSummary, ExperimentInputs
from rag_evaluator.config import ExperimentConfig
from rag_evaluator.datasets.loader import load_dataset_from_config
from rag_evaluator.ingestion.chunkers import SourceDocument
from rag_evaluator.schemas import EvalSample


def load_experiment_inputs(experiment: ExperimentConfig) -> ExperimentInputs:
    """
    Load normalized samples and reconstruct source documents for an experiment.
    """

    all_samples: list[EvalSample] = []
    dataset_summaries: list[DatasetLoadSummary] = []

    for dataset in experiment.datasets:
        dataset_samples = load_dataset_from_config(dataset)
        all_samples.extend(dataset_samples)
        dataset_summaries.append(
            DatasetLoadSummary(
                dataset_name=dataset.name,
                sample_count=len(dataset_samples),
            )
        )

    documents = build_source_documents(all_samples)
    return ExperimentInputs(
        experiment=experiment,
        samples=all_samples,
        documents=documents,
        datasets=dataset_summaries,
    )


def build_source_documents(samples: list[EvalSample]) -> list[SourceDocument]:
    """
    Reconstruct source documents from evidence spans or fallback reference answers.
    """

    documents_by_id: dict[str, dict[str, object]] = {}

    for sample in samples:
        for evidence_span in sample.evidence_spans:
            if not evidence_span.text:
                continue

            entry = documents_by_id.setdefault(
                evidence_span.document_id,
                {
                    "parts": [],
                    "seen_texts": set(),
                    "metadata": {
                        "source": "evidence_span",
                        "source_dataset": sample.source_dataset,
                    },
                },
            )
            seen_texts = entry["seen_texts"]
            parts = entry["parts"]
            if not isinstance(seen_texts, set) or not isinstance(parts, list):
                raise TypeError("Invalid source document accumulator state.")

            if evidence_span.text in seen_texts:
                continue

            seen_texts.add(evidence_span.text)
            parts.append(evidence_span.text)

        if sample.reference_answer and sample.sample_id not in documents_by_id:
            documents_by_id[sample.sample_id] = {
                "parts": [sample.reference_answer],
                "seen_texts": {sample.reference_answer},
                "metadata": {
                    "source": "reference_answer",
                    "source_dataset": sample.source_dataset,
                },
            }

    documents: list[SourceDocument] = []
    for document_id, entry in documents_by_id.items():
        parts = entry["parts"]
        metadata = entry["metadata"]
        if not isinstance(parts, list) or not isinstance(metadata, dict):
            raise TypeError("Invalid source document entry.")

        text = "\n\n".join(part for part in parts if isinstance(part, str) and part.strip())
        if not text:
            continue

        documents.append(
            SourceDocument(
                document_id=document_id,
                text=text,
                metadata=metadata,
            )
        )

    if not documents:
        raise ValueError(
            "run-experiment requires datasets with evidence_spans text or "
            "reference_answer so source documents can be reconstructed."
        )

    return documents
