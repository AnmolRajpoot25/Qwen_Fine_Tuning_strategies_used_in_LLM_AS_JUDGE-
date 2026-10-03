
import copy


def swap_answers(example):

    swapped = copy.deepcopy(example)

    swapped["id"] = (
        f'{example["id"]}_swapped'
    )

    swapped["answer_a"] = example["answer_b"]

    swapped["answer_b"] = example["answer_a"]

    swapped["evaluation"] = {
        "A": example["evaluation"]["B"],
        "B": example["evaluation"]["A"]
    }

    return swapped


def augment_with_position_swap(dataset):

    augmented = []

    for example in dataset:

        augmented.append(example)

        augmented.append(
            swap_answers(example)
        )

    return augmented
