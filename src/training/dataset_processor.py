
import json
import random
from pathlib import Path
from collections import Counter

from src.training.schema import validate_training_example
from src.training.augment import augment_with_position_swap
from src.training.formatter import format_dataset


class DatasetProcessor:

    def __init__(
        self,
        raw_dir,
        output_dir,
        validation_ratio=0.1,
        random_seed=42
    ):

        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)

        self.validation_ratio = validation_ratio
        self.random_seed = random_seed

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def load_raw_data(self):

        dataset = []

        json_files = list(
            self.raw_dir.glob("*.json")
        )

        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in {self.raw_dir}"
            )

        for file_path in json_files:

            print(
                f"Loading: {file_path.name}"
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

                if isinstance(data, list):
                    dataset.extend(data)

                elif isinstance(data, dict):
                    dataset.append(data)

                else:
                    print(
                        f"Skipping invalid file: "
                        f"{file_path.name}"
                    )

        return dataset

    def validate_dataset(self, dataset):

        valid_examples = []
        invalid_examples = []

        for example in dataset:

            if validate_training_example(example):

                valid_examples.append(example)

            else:

                invalid_examples.append({
                    "id": example.get(
                        "id",
                        "unknown"
                    ),
                    "reason": "Schema validation failed"
                })

        return (
            valid_examples,
            invalid_examples
        )

    def remove_duplicates(self, dataset):

        unique_examples = []
        seen = set()

        for example in dataset:

            key = (
                example["problem"].strip().lower(),
                example["answer_a"].strip().lower(),
                example["answer_b"].strip().lower()
            )

            if key not in seen:

                seen.add(key)

                unique_examples.append(example)

        return unique_examples

    def split_dataset(self, dataset):

        random.seed(
            self.random_seed
        )

        shuffled = dataset.copy()

        random.shuffle(shuffled)

        validation_size = int(
            len(shuffled)
            * self.validation_ratio
        )

        validation_data = shuffled[
            :validation_size
        ]

        train_data = shuffled[
            validation_size:
        ]

        return (
            train_data,
            validation_data
        )

    def save_jsonl(
        self,
        dataset,
        file_path
    ):

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as file:

            for example in dataset:

                file.write(
                    json.dumps(
                        example,
                        ensure_ascii=False
                    )
                    + "\n"
                )

    def generate_statistics(
        self,
        dataset
    ):

        categories = Counter(
            example["category"]
            for example in dataset
        )

        difficulties = Counter(
            example["difficulty"]
            for example in dataset
        )

        return {
            "total_examples": len(dataset),

            "categories": dict(categories),

            "difficulties": dict(difficulties)
        }

    def process(self):

        print("=" * 60)
        print("LOADING RAW DATA")
        print("=" * 60)

        raw_data = self.load_raw_data()

        print(
            f"Raw examples: {len(raw_data)}"
        )

        print("\n" + "=" * 60)
        print("VALIDATING DATA")
        print("=" * 60)

        valid_data, invalid_data = (
            self.validate_dataset(
                raw_data
            )
        )

        print(
            f"Valid examples: {len(valid_data)}"
        )

        print(
            f"Invalid examples: {len(invalid_data)}"
        )

        print("\n" + "=" * 60)
        print("REMOVING DUPLICATES")
        print("=" * 60)

        unique_data = self.remove_duplicates(
            valid_data
        )

        print(
            f"Unique examples: {len(unique_data)}"
        )

        print(
            f"Duplicates removed: "
            f"{len(valid_data) - len(unique_data)}"
        )

        print("\n" + "=" * 60)
        print("SPLITTING DATASET")
        print("=" * 60)

        # IMPORTANT:
        # Split BEFORE augmentation to prevent
        # original and swapped versions of the same
        # example appearing in train and validation.

        train_raw, validation_raw = (
            self.split_dataset(
                unique_data
            )
        )

        print(
            f"Train raw: {len(train_raw)}"
        )

        print(
            f"Validation raw: {len(validation_raw)}"
        )

        print("\n" + "=" * 60)
        print("AUGMENTING TRAINING DATA")
        print("=" * 60)

        train_augmented = (
            augment_with_position_swap(
                train_raw
            )
        )

        validation_augmented = (
            augment_with_position_swap(
                validation_raw
            )
        )

        print(
            f"Train after augmentation: "
            f"{len(train_augmented)}"
        )

        print(
            f"Validation after augmentation: "
            f"{len(validation_augmented)}"
        )

        print("\n" + "=" * 60)
        print("FORMATTING FOR QWEN")
        print("=" * 60)

        train_formatted = format_dataset(
            train_augmented
        )

        validation_formatted = format_dataset(
            validation_augmented
        )

        train_path = (
            self.output_dir
            / "train.jsonl"
        )

        validation_path = (
            self.output_dir
            / "validation.jsonl"
        )

        self.save_jsonl(
            train_formatted,
            train_path
        )

        self.save_jsonl(
            validation_formatted,
            validation_path
        )

        statistics = {
            "raw": self.generate_statistics(
                raw_data
            ),

            "clean": self.generate_statistics(
                unique_data
            ),

            "train": self.generate_statistics(
                train_augmented
            ),

            "validation": self.generate_statistics(
                validation_augmented
            ),

            "invalid_examples": len(
                invalid_data
            ),

            "duplicates_removed": (
                len(valid_data)
                - len(unique_data)
            )
        }

        statistics_path = (
            self.output_dir
            / "statistics.json"
        )

        with open(
            statistics_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                statistics,
                file,
                indent=2
            )

        print("\n" + "=" * 60)
        print("DATASET PROCESSING COMPLETE")
        print("=" * 60)

        print(
            f"Train file: {train_path}"
        )

        print(
            f"Validation file: {validation_path}"
        )

        print(
            f"Statistics: {statistics_path}"
        )

        return {
            "train_path": str(train_path),
            "validation_path": str(
                validation_path
            ),
            "statistics": statistics
        }
