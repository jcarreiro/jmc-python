import unittest

from jmc.algorithms.puzzles import *

class TestListThreeIntSumToZero(unittest.TestCase):
    def setUp(self):
        self.A = [0, 1, 2, -2, -1]
        self.expected_solutions = [
            [0, 0, 0],
            [0, 1, 4],
            [0, 2, 3],
            [1, 1, 3],
            [2, 4, 4],
        ]

    def check(self, solutions, expected_solutions):
        for t in solutions:
            self.assertIn(sorted(t), expected_solutions)

    def test_naive(self):
        solutions = list_three_int_sum_to_zero_naive(self.A)
        self.check(solutions, self.expected_solutions)

    def test_hashtable(self):
        solutions = list_three_int_sum_to_zero_hashtable(self.A)
        self.check(solutions, self.expected_solutions)

    def test_optimal(self):
        # The set of expected solutions is different here, because the function
        # sorts the list; when sorted, it looks like [-2, -1, 0, 1, 2].
        expected_solutions = [
            [0, 2, 4],
            [1, 2, 3],
            [2, 2, 2],
            [0, 3, 3],
            [1, 1, 4],
        ]
        solutions = list_three_int_sum_to_zero(self.A)
        self.check(solutions, expected_solutions)

class TestBalanceParens(unittest.TestCase):
    def test(self):
        self.assertEqual(balance_parens(list("()")), list("()"))
        self.assertEqual(balance_parens(list("a(b)c)")), list("a(b)c"))
        self.assertEqual(balance_parens(list(")(")), list(""))
        self.assertEqual(balance_parens(list("(((((")), list(""))
        self.assertEqual(balance_parens(list("(()()(")), list("(())")) # or list("()()")
        self.assertEqual(balance_parens(list(")(())(")), list("(())"))
