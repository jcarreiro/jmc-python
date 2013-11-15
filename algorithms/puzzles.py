from jmc.decorators import countcalls, memoized

# Algorithmic puzzles!

# Missing Integer
# ---------------
# Given an array, A[1..N], containing all of the numbers from 0 to N, except for
# one, find the missing integer in linear time. Additionally, you may not access
# a complete integer; the elements of A are stored in binary, and the only
# operation supported is fetching the j-th bit of A[i], in constant time.
#
# Ref: CLRS (pages?)
def missing_integer(A, j=0, bits=[]):
    pass


# Making Change
# -------------
# Given an amount, N, find the total number of ways to make change for M, using
# pennies, nickels, dimes, and quarters.
#
# We can break this down into two subproblems:
#
#   1. The number of ways to make change for N, after having used a coin of the
#      largest denomination.
#
#   2. The number of ways to make change for N, without using any coins of the
#      largest denomination.
#
# Solving the subproblems recursively and combining the answers gives us the
# answer to the problem.
@memoized
@countcalls
def make_change(n, d):
    if n == 0:
        return 1
    elif n < 0 or len(d) == 0:
        return 0
    else:
        return make_change(n - d[-1], d) + make_change(n, d[:-1])

# Adding Up
# ---------
# Like making change, but we count the ways to sum the numbers (1, N-1) to make
# N.
#
# Interestingly, these numbers are the partition numbers (Sloane's A000041),
# with the only difference being that we don't count 0 + N as a partition of
# N.
#
# See also http://mathworld.wolfram.com/PartitionFunctionP.html.
@memoized
@countcalls
def number_of_partitions(n):
    def count_partitions(n, d):
        if n == 0:
            return 1
        elif n < 0 or len(d) == 0:
            return 0
        else:
            return count_partitions(n - d[-1], d) + count_partitions(n, d[:-1])
    return count_partitions(n, range(1, n))
