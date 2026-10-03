
import json

from src.judge.prompts import JUDGE_SYSTEM_PROMPT


def format_training_example(example):

    user_content = f"""
PROBLEM:

{example["problem"]}

ANSWER A:

{example["answer_a"]}

ANSWER B:

{example["answer_b"]}

Evaluate both answers independently.

Return ONLY the required JSON.
"""

    assistant_content = json.dumps(
        example["evaluation"],
        ensure_ascii=False
    )

    return {
        "messages": [
            {
                "role": "system",
                "content": JUDGE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content.strip()
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    }


def format_dataset(dataset):

    formatted = []

    for example in dataset:

        formatted.append(
            format_training_example(
                example
            )
        )

    return formatted
