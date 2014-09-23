#!/usr/bin/python
#
# Solution to exercise 1.2 from the book _Essential Algorithms_.

import math
import numpy as np
import matplotlib.pyplot as plt


def print_table():
    # number of algorithm 'steps' per second
    rate = 1E6

    # run-times in seconds: 1 sec, 1 min, 1 hour, 1 day, 1 week, 1 year
    periods = [1, 60, 3600, 86400, 604800, 31536000]

    # orders of growth: log n, sqrt n, n, n^2, 2^n, n!
    def inv_fact(s):
        n = 1
        while s > math.factorial(n):
            n += 1
        return n

    fns = [
        ['log n',  lambda s: 2 ** s],         # log
        ['sqrt n', lambda s: s ** 2],         # sqrt
        ['n',      lambda s: s],              # linear
        ['n^2',    lambda s: math.sqrt(s)],   # quadratic
        ['2^n',    lambda s: math.log(s, 2)], # exponential
        ['n!',     inv_fact],                 # factorial
    ]

    print '{0:8s}'.format(''),
    for t in periods:
        print '{0:12d}'.format(t),
    print

    for label, fn in fns:
        print '{0:8s}'.format(label),
        for t in periods:
            try:
                print '{0:12g}'.format(fn(t * rate)),
            except OverflowError:
                print '  <overflow>',
        print

if __name__ == '__main__':
    print_table()
