from __future__ import annotations

from rag_evaluator.datasets.config import DatasetConfig
from rag_evaluator.datasets.normalizers.base import DatasetNormalizer
from rag_evaluator.datasets.normalizers.code.codesearchnet import CodeSearchNetNormalizer
from rag_evaluator.datasets.normalizers.code.conala import ConalaNormalizer
from rag_evaluator.datasets.normalizers.domain.cuad import CuadNormalizer
from rag_evaluator.datasets.normalizers.domain.finqa import FinQANormalizer
from rag_evaluator.datasets.normalizers.domain.legalbench import LegalBenchNormalizer
from rag_evaluator.datasets.normalizers.domain.med_qa import MedQANormalizer
from rag_evaluator.datasets.normalizers.domain.pubmedqa import PubMedQANormalizer
from rag_evaluator.datasets.normalizers.general.natural_questions import (
    NaturalQuestionsNormalizer,
)
from rag_evaluator.datasets.normalizers.general.squad_v2 import SquadV2Normalizer
from rag_evaluator.datasets.normalizers.general.trivia_qa import TriviaQANormalizer
from rag_evaluator.datasets.normalizers.long_context.narrativeqa import (
    NarrativeQANormalizer,
)
from rag_evaluator.datasets.normalizers.long_context.qasper import QasperNormalizer
from rag_evaluator.datasets.normalizers.long_context.quality import QualityNormalizer
from rag_evaluator.datasets.normalizers.multihop.hotpot_qa import HotpotQANormalizer
from rag_evaluator.datasets.normalizers.multihop.musique import MusiqueNormalizer
from rag_evaluator.datasets.normalizers.multihop.two_wiki_multihop_qa import (
    TwoWikiMultiHopQANormalizer,
)
from rag_evaluator.datasets.normalizers.retrieval.beir import BeirNormalizer
from rag_evaluator.datasets.normalizers.robustness.rgb import RGBNormalizer
from rag_evaluator.datasets.normalizers.synthetic.ragas import RagasNormalizer

NORMALIZER_REGISTRY: dict[str, type[DatasetNormalizer]] = {
    "natural_questions": NaturalQuestionsNormalizer,
    "trivia_qa": TriviaQANormalizer,
    "squad_v2": SquadV2Normalizer,
    "hotpot_qa": HotpotQANormalizer,
    "musique": MusiqueNormalizer,
    "two_wiki_multihop_qa": TwoWikiMultiHopQANormalizer,
    "qasper": QasperNormalizer,
    "narrativeqa": NarrativeQANormalizer,
    "quality": QualityNormalizer,
    "cuad": CuadNormalizer,
    "legalbench": LegalBenchNormalizer,
    "med_qa": MedQANormalizer,
    "pubmedqa": PubMedQANormalizer,
    "finqa": FinQANormalizer,
    "conala": ConalaNormalizer,
    "codesearchnet": CodeSearchNetNormalizer,
    "beir": BeirNormalizer,
    "rgb": RGBNormalizer,
    "ragas": RagasNormalizer,
}


def build_normalizer(dataset_key: str, config: DatasetConfig) -> DatasetNormalizer:
    """Build a dataset normalizer by catalog normalizer key."""

    try:
        normalizer_cls = NORMALIZER_REGISTRY[dataset_key]
    except KeyError as exc:
        known = ", ".join(sorted(NORMALIZER_REGISTRY))
        raise ValueError(f"No normalizer registered for '{dataset_key}'. Known: {known}") from exc

    return normalizer_cls(config)
