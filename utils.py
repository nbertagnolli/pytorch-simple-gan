from typing import List, Tuple
import numpy as np
import math


def create_binary_list_from_int(number: int) -> List[int]:
    """Creates a list of the binary representation of a positive integer

    Args:
        number: An integer

    Returns:
        The binary representation of the provided positive integer number as a list.
    """
    if number < 0 or type(number) is not int:
        raise ValueError("Only Positive integers are allowed")

    return [int(x) for x in list(bin(number))[2:]]


def generate_even_data(
    max_int: int, batch_size: int = 16
) -> Tuple[List[int], List[List[int]]]:
    """An infinite data generator which yields

    Args:
        max_int: The maximum input integer value
        batch_size: The size of the training batch.

    Returns:
        A Tuple with the labels and the input data.
        labels:
        data:
    """

    # Get the number of binary places needed to represent the maximum number
    max_length = int(math.log(max_int, 2))

    # Sample batch_size number of integers in range 0-max_int
    sampled_integers = np.random.randint(0, int(max_int / 2), batch_size)

    # create a list of labels all ones because all numbers are even
    labels = [1] * batch_size

    # Generate a list of binary numbers for training.
    data = [create_binary_list_from_int(int(x * 2)) for x in sampled_integers]
    data = [([0] * (max_length - len(x))) + x for x in data]

    return labels, data


def convert_float_matrix_to_int_list(
    float_matrix: np.array, threshold: float = 0.5
) -> List[int]:
    """Converts generated output in binary list form to a list of integers

    Args:
        float_matrix: A matrix of values between 0 and 1 which we want to threshold and convert to
            integers
        threshold: The cutoff value for 0 and 1 thresholding.

    Returns:
        A list of integers.
    """
    return [
        int("".join([str(int(y)) for y in x]), 2) for x in float_matrix >= threshold
    ]
