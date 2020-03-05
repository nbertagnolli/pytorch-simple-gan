import math
from typing import List
import unittest

from ddt import data, ddt, unpack
import numpy as np

from utils import (
    create_binary_list_from_int,
    generate_even_data,
    convert_float_matrix_to_int_list,
)


@ddt
class UtilsTest(unittest.TestCase):
    @data(
        (1, [1], "test single odd"),
        (0, [0], "test zero"),
        (7, [1, 1, 1], "test small prime"),
        (128, [1, 0, 0, 0, 0, 0, 0, 0], "test large even"),
        (129, [1, 0, 0, 0, 0, 0, 0, 1], "test large odd"),
    )
    @unpack
    def test_create_binary_list_from_int(
        self, number: int, expected: List[int], test_description: str
    ):
        output = create_binary_list_from_int(number)
        self.assertListEqual(expected, output, test_description)

    @data(
        (2, 2, "test small input"),
        (128, 2, "test standard input"),
        (128, 16, "test standard input big batch size"),
        (1024, 16, "test big input"),
    )
    @unpack
    def test_generate_even_data(
        self, max_int: int, batch_size: int, test_description: str
    ):
        labels, output = generate_even_data(max_int, batch_size=batch_size)
        self.assertEqual(len(labels), batch_size)
        self.assertEqual(len(output), batch_size)
        for binary_num in output:
            self.assertEqual(binary_num[-1], 0, test_description)
            self.assertEqual(len(binary_num), math.log(max_int, 2), test_description)

    @data(([[0.6, 0.2], [0.5, 0.5]], [2, 3], "test two values"))
    @unpack
    def test_convert_float_matrix_to_int_list(
        self, matrix: List[List[float]], expected: List[int], test_description: str
    ):
        matrix = np.array(matrix)
        output = convert_float_matrix_to_int_list(np.array(matrix))
        self.assertListEqual(expected, output, test_description)


if __name__ == "__main__":
    unittest.main()
