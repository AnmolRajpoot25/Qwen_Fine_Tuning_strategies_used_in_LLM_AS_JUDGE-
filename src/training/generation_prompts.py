
DATA_GENERATION_SYSTEM_PROMPT = """
You are an expert benchmark dataset creator for training an LLM judge.

Your task is to generate high-quality technical question and answer comparison examples.

The dataset will train another model to evaluate two technical answers.

IMPORTANT REQUIREMENTS:

1. Problems must be technically realistic.
2. Avoid trivial factual questions whenever possible.
3. Answers should often be close in quality.
4. Incorrect answers must be plausible.
5. Include subtle reasoning errors.
6. Include edge-case mistakes.
7. Include complexity mistakes where relevant.
8. Do not make the incorrect answer obviously absurd.
9. Score each answer independently.
10. Scores must be integers from 0 to 10.

EVALUATION CRITERIA:

correctness:
Technical accuracy.

relevance:
How directly the answer addresses the problem.

completeness:
Whether important information is missing.

reasoning:
Quality and logical validity of explanation.

clarity:
How clearly the answer is communicated.

The output must be valid JSON only.

Do not include markdown.
Do not include explanations outside JSON.
"""


def build_generation_prompt(
    category,
    topic,
    difficulty,
    failure_type,
    variant_a,
    variant_b
):

    return f"""
Generate ONE high-quality LLM judge training example.

CATEGORY:
{category}

TOPIC:
{topic}

DIFFICULTY:
{difficulty}

PRIMARY FAILURE TYPE:
{failure_type}

ANSWER A QUALITY:
{variant_a}

ANSWER B QUALITY:
{variant_b}

The example must contain:

1. A realistic technical problem.
2. Answer A.
3. Answer B.
4. Independent scores for both answers.
5. Concise expert feedback.

Return exactly this JSON schema:

{{
    "problem": "...",

    "answer_a": "...",

    "answer_b": "...",

    "evaluation": {{
        "A": {{
            "correctness": 0,
            "relevance": 0,
            "completeness": 0,
            "reasoning": 0,
            "clarity": 0,
            "feedback": "..."
        }},

        "B": {{
            "correctness": 0,
            "relevance": 0,
            "completeness": 0,
            "reasoning": 0,
            "clarity": 0,
            "feedback": "..."
        }}
    }}
}}

Ensure the score differences reflect the actual technical quality.

Avoid giving perfect 10 scores to every good answer.

Make the comparison challenging enough to test an expert technical evaluator.
"""
