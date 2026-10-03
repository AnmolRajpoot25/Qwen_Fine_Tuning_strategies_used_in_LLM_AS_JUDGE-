
import sys

from configs.dataset_config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    VALIDATION_RATIO,
    RANDOM_SEED
)

from src.training.dataset_processor import (
    DatasetProcessor
)


def main():

    processor = DatasetProcessor(

        raw_dir=RAW_DATA_DIR,

        output_dir=PROCESSED_DATA_DIR,

        validation_ratio=VALIDATION_RATIO,

        random_seed=RANDOM_SEED
    )

    result = processor.process()

    print("\nFINAL STATISTICS")

    for split, stats in (
        result["statistics"].items()
    ):

        if isinstance(stats, dict):

            print(
                f"\n{split.upper()}"
            )

            for key, value in stats.items():

                print(
                    f"{key}: {value}"
                )


if __name__ == "__main__":

    main()
