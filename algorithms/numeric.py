# ------------------------------------------------------------------------------
# Numeric algorithms
#
# Copyright (c) 2013, Jason M. Carreiro.
#
# See LICENSE file for license information.
# ------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import random

# Shuffles a sequence, in place. We do this by choosing an index with which to
# swap each element uniformly at random.
#
# Yes, I know about random.shuffle, but I'm reading _Essential Algorithms_ and
# I want to actually type up and run all of the algorithms from the book as an
# aid to my understanding.
def shuffle(A):
    """Permutes an array, in place."""
    for i in range(0, len(A)):
        j = random.randint(i, len(A) - 1)
        t = A[i]
        A[i] = A[j]
        A[j] = t

def test_shuffle():
    A = range(0, 10)
    results = []
    for i in xrange(0, 1000):
        results.append(shuffle(A))
