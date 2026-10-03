
ANSWER_VARIANTS = {

    "excellent": {
        "description": (
            "Fully correct answer with accurate reasoning, "
            "appropriate details, edge cases where relevant, "
            "and correct complexity or trade-offs."
        )
    },


    "correct_but_incomplete": {
        "description": (
            "Core answer is correct but misses important details, "
            "edge cases, complexity analysis, limitations, "
            "or implementation considerations."
        )
    },


    "partially_correct": {
        "description": (
            "Contains some correct concepts but has meaningful "
            "technical omissions or incorrect reasoning."
        )
    },


    "subtle_error": {
        "description": (
            "Appears plausible and mostly correct but contains "
            "a subtle technical error, incorrect assumption, "
            "boundary condition failure, or complexity mistake."
        )
    },


    "incorrect": {
        "description": (
            "Directly incorrect answer with flawed technical "
            "understanding or invalid reasoning."
        )
    },


    "hallucinated": {
        "description": (
            "Confident answer containing fabricated facts, APIs, "
            "features, algorithms, or unsupported claims."
        )
    }
}
