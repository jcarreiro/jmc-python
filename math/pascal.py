# Generates Pascal's Triangle
#
#           1
#          1 1
#         1 2 1
#        1 3 3 1
#
# etc...

from numeric import *

# This is an iterative solution.
def pascals_triangle(n):
    for r in range(0, n):
        for c in range(0, r+1):
            print n_choose_k(r, c),
        print

# This is a tree recursive solution.
# Each node t_r,c = t_r-1,c-1 + t_r-1,c.
def t(r, c):
    if c == 0 or c == r:
        return 1
    return t(r - 1, c - 1) + t(r - 1, c)

def pascals_triangle_tree(n):
    for r in range(0, n):
        for c in range(0, r+1):
            print t(r, c),
        print
