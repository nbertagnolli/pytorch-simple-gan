import math
from typing import List
import unittest

from ddt import data, ddt, unpack
import torch

from utils import convert_float_matrix_to_int_list
from train import train


@ddt
class TrainTest(unittest.TestCase):
    @data(
        (128, 16, 500, 0.001, "Test reasonable parameters"),
        (256, 16, 500, 0.001, "Test reasonable parameters"),
    )
    @unpack
    def test_train(
        self,
        max_int: int,
        batch_size: int,
        training_steps: int,
        learning_rate: float,
        test_description: str,
    ):
        input_length = int(math.log(max_int, 2))
        generator, discriminator = train(
            max_int=max_int,
            batch_size=batch_size,
            training_steps=training_steps,
            learning_rate=learning_rate,
            print_output_every_n_steps=1000000,
        )
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        generated_data = generator(noise)
        for num in convert_float_matrix_to_int_list(generated_data):
            self.assertEqual(num % 2, 0, test_description)


if __name__ == "__main__":
    unittest.main()
