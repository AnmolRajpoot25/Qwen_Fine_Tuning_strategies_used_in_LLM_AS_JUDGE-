
from typing import Dict, Any


REQUIRED_CRITERIA = {
    "correctness",
    "relevance",
    "completeness",
    "reasoning",
    "clarity"
}


def validate_score(value):

    if not isinstance(value, int):
        return False

    return 0 <= value <= 10


def validate_candidate(candidate):

    if not isinstance(candidate, dict):
        return False

    required = REQUIRED_CRITERIA | {"feedback"}

    if not required.issubset(
        candidate.keys()
    ):
        return False

    for criterion in REQUIRED_CRITERIA:

        if not validate_score(
            candidate[criterion]
        ):
            return False

    if not isinstance(
        candidate["feedback"],
        str
    ):
        return False

    return True


def validate_training_example(example):

    required_fields = {
        "id",
        "category",
        "difficulty",
        "problem",
        "answer_a",
        "answer_b",
        "evaluation"
    }

    if not required_fields.issubset(
        example.keys()
    ):
        return False

    evaluation = example["evaluation"]

    if not isinstance(evaluation, dict):
        return False

    if "A" not in evaluation:
        return False

    if "B" not in evaluation:
        return False

    if not validate_candidate(
        evaluation["A"]
    ):
        return False

    if not validate_candidate(
        evaluation["B"]
    ):
        return False

    return True
