#!/usr/bin/python
#
# Solution to exercise 1.2 from the book _Essential Algorithms_.

import math
import numpy as np
import matplotlib.pyplot as plt

# Orders of growth. For some reason, matplotlib is happy when these are defined
# as functions, but not happy if I try to give it lambdas?
def inv_log(s):
    return 2 ** s

def inv_sqrt(s):
    return s ** 2

def inv_linear(s):
    return s

def inv_quad(s):
    return math.sqrt(s)

def inv_exp(s):
    return math.log(s, 2)

def inv_fact(s):
    n = 1
    while s > math.factorial(n):
        n += 1
    return n

def make_table():
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

    table = [[''] + periods]
    for label, fn in fns:
        row = [label]
        for t in periods:
            try:
                row.append(fn(t * rate))
            except OverflowError:
                row.append('  <overflow>')
        table.append(row)
    return table

def print_table_text(table):
    for row in table:
        for col in row:
            if type(col) == int:
                print '{0:12d}'.format(col),
            elif type(col) == float:
                print '{0:12g}'.format(col),
            else:
                print '{0:12s}'.format(col),
        print
    print

def print_table_html(table):
    print '<table>'
    print '  <th>'

    # assume the first row is a header row
    for col in table[0]:
        print '    <td>{0}</td>'.format(col)
    print '  </th>'

    # now we print the data
    for row in table[1:]:
        print '  <tr>'
        for col in row:
            print '    <td>{0}</td>'.format(col)
        print '  </tr>'
    print '</table>'

# Show the growth in the problem size, for each function, as a function of time
# in seconds.
def plot_results():
    t = np.arange(1.0, 100.0) # 100 s
    fns = [
        ['log n',  inv_log],
        ['sqrt n', inv_sqrt],
        ['n',      inv_linear],
        ['n^2',    inv_quad],
        ['2^n',    inv_exp],
        ['n!',     inv_fact],
    ]
    for label, fn in fns:
        plt.plot(t, [fn(x) for x in t], label=label)
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    print_table_html(make_table())
