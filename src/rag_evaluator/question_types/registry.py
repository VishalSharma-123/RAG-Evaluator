from __future__ import annotations

from rag_evaluator.question_types.base import QuestionTypeRule
from rag_evaluator.question_types.rules import (
    AbstractiveRule,
    AdversarialRule,
    ComparativeRule,
    FactoidRule,
    MultiHopRule,
    UnanswerableRule,
)
from rag_evaluator.schemas import QuestionType

QUESTION_TYPE_RULES: dict[QuestionType, QuestionTypeRule] = {
    QuestionType.FACTOID: FactoidRule(),
    QuestionType.MULTI_HOP: MultiHopRule(),
    QuestionType.ABSTRACTIVE: AbstractiveRule(),
    QuestionType.ADVERSARIAL: AdversarialRule(),
    QuestionType.COMPARATIVE: ComparativeRule(),
    QuestionType.UNANSWERABLE: UnanswerableRule(),
}

def get_question_type_rule(question_type: QuestionType) -> QuestionTypeRule:
    """
    Return the registered rule for one question type.
    :param question_type:
    :return:
    """
    return QUESTION_TYPE_RULES[question_type]

def get_question_type_rules(
        question_types: list[QuestionType] | None = None,
) -> dict[QuestionType, QuestionTypeRule]:
    """
    Return all registered rules or the subset for the requested question types.
    :param question_types:
    :return:
    """
    if question_types is None:
        return dict(QUESTION_TYPE_RULES)
    return {
        question_type: QUESTION_TYPE_RULES[question_type]
        for question_type in question_types
    }