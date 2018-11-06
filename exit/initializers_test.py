# python -m unittest discover -s ./ -p '*_test.py'

from .initializers import Constant
import unittest
import numpy as np


class TestConstant(unittest.TestCase):
    def test_value(self):
        shape = (2, 3)
        constant_function = Constant(3)
        constant = constant_function(shape[0], shape[1])
        self.assertEqual(constant.shape, shape)
        self.assertEqual(constant.tolist(), [[3,3,3],[3,3,3]])
