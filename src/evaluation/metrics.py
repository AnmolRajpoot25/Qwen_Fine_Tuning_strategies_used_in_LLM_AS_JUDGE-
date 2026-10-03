from collections import defaultdict


class BenchmarkMetrics:

    @staticmethod
    def calculate(results):

        total = len(results)

        correct = sum(
            1
            for result in results
            if result["correct"]
        )

        accuracy = (
            correct / total
            if total
            else 0
        )

        category_stats = defaultdict(
            lambda: {
                "total": 0,
                "correct": 0
            }
        )

        for result in results:

            category = result["category"]

            category_stats[category]["total"] += 1

            if result["correct"]:
                category_stats[category]["correct"] += 1

        for category in category_stats:

            stats = category_stats[category]

            stats["accuracy"] = (
                stats["correct"]
                / stats["total"]
            )

        return {
            "total": total,
            "correct": correct,
            "accuracy": round(
                accuracy,
                4
            ),
            "categories": dict(category_stats)
        }