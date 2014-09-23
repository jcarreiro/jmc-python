# Random utility functions.

import itertools

# Flatten a list of lists, e.g. [[1, 2], [3, 4]] => [1, 2, 3, 4].
def flatten(l):
    return list(itertools.chain.from_iterable(l)
