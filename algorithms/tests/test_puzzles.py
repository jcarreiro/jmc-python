import unittest

from jmc.algorithms.puzzles import *

class TestMaxPathInTree(unittest.TestCase):
    def test_simple(self):
        # Given:
        #
        #       1
        #      / \
        #     2   3
        #    / \   \
        #   4   5   6
        #
        # The max path is (5, 2, 1, 3, 6), which has the value 17.
        self.assertEqual(max_path_in_tree(BNode(1,
                                                BNode(2,
                                                      BNode(4, None, None),
                                                      BNode(5, None, None)),
                                                BNode(3,
                                                      None,
                                                      BNode(6, None, None)))),
                                          17)

    def test_empty(self):
        self.assertEqual(max_path_in_tree(None), 0)

    def test_single(self):
        self.assertEqual(max_path_in_tree(BNode(1, None, None)), 1)

    def test_max_path_not_through_root(self):
        self.assertEqual(max_path_in_tree(BNode(1,
                                                BNode(2,
                                                      BNode(400, None, None),
                                                      BNode(500, None, None)),
                                                BNode(3,
                                                      None,
                                                      BNode(6, None, None)))),
                         902)

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

class TestDeepCopyList(unittest.TestCase):
    def testEmpty(self):
        self.assertEqual(deep_copy_list(None), None)

    def testSingleNode(self):
        l = ExtraNode(1)
        m = deep_copy_list(l)
        self.assertNotNone(m)
        # todo: more assertions

if __name__ == '__main__':
    unittest.main()
