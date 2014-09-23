# ------------------------------------------------------------------------------
# Algorithmic puzzles!
#
# Copyright (c) 2013, Jason M. Carreiro.
#
# See LICENSE file for license information.
#
# todo: It would be nice to be able to view the call graph for the
#       tree-recursive problems using pylab.
#
# problem todos:
#   - add two binary strings
#   - anagram printing
#   - return index of max value from array (uniformly at random)
# ------------------------------------------------------------------------------

from __future__ import absolute_import

import random

from jmc.decorators import countcalls, memoized

# -----------------------------------------------------------------------------
# Permutations
#
# Question
# --------
# Given a string, "generate" all of the permutations of the string.
#
# Solution
# --------
# This is a straightforward tree-recursive problem with a simple solution.
# -----------------------------------------------------------------------------
def permutations(s):
    def permutations_helper(s, t):
        if len(s) == 0:
            yield t
        else:
            for c in s:
                for p in permutations_helper(s - set(c), t + c):
                    yield p
    for p in permutations_helper(set(s), ''):
        yield p

# More than one candidate has given me this "solution". Since it only prints
# n^2 strings and there are n! permutations, it can't possibly be correct...
def permutations_wrong(s):
    for i in xrange(len(s)):
        c = s[i]
        t = s[0:i] + s[i + 1:]
        for j in xrange(len(t) + 1):
            yield t[0:j] + c + t[j:]

# This version returns a list -- which is bad idea, since the number of list
# entries grows as O(n!).
def permutations_list(s):
    if len(s) == 1:
        return [s]
    l = []
    for i in xrange(len(s)):
        for p in permutations_list(s[0:i] + s[i+1:]):
            l += [s[i] + p]
    return l

# ------------------------------------------------------------------------------
# Towers of Hanoi
#
# Question
# --------
#
# Solution
# --------
#
# ------------------------------------------------------------------------------
def towers_of_hanoi():
    pass

# -----------------------------------------------------------------------------
# Tree to (Doubly-linked) List
# -----------------------------------------------------------------------------
#
# Question
# --------
# Given a binary tree, convert the tree into a list, in place.
#
# Solution
# --------
# A recursive solution in C is easy ... but how do we do it in Python?
#
# Ref: Interview question used at Facebook.
# -----------------------------------------------------------------------------
def tree_to_list(t):
    pass

# ------------------------------------------------------------------------------
# Missing Integer and Variations
# ------------------------------------------------------------------------------
#
# Question
# --------
# Given an array, A[1..N], containing all of the numbers from 0 to N, except for
# one, find the missing integer in linear time. Additionally, you may not access
# a complete integer; the elements of A are stored in binary, and the only
# operation supported is fetching the j-th bit of A[i], in constant time.
#
# Solution
# --------
# The solution is to recursively divide the array into two parts based on the
# j-th bit, one of which contains all of the integers where the bit is set, the
# other of which contains all of the integers where it isn't. If the array
# contained every number from 0 to N, then we would expect both parts to be the
# same size, so the smaller half must 'contain' the missing integer. Hence the
# j-th bit of the missing integer must be equal to a 1 if the '1's part is
# smaller or a zero otherwise.
#
# This insight alone is only enough to get us to an O(N lg N) solution, if we
# examine the entire array for each possible bit position. However we can also
# eliminate half the remaining entries at each step, since we only need to
# recurse on the half that 'contains' the missing integer. This leads us to the
# recurrence:
#
#   T(N) = T(N/2) + \Theta(N)
#
# The solution to which is O(N) by the master method:
#
# If T(n) = aT(n/b) + f(n), then
#
#   (1) If f(n) = O(n^{log_b a - \epsilon}) for some constant \epsilon > 0,
#       then T(n) = \Theta(n^{log_b a})
#   (2) If f(n) = \Theta(n^{log_b a}, then T(n) = \Theta(n^{log_b a} lg n)
#   (3) If f(n) = \Omega(n^{log_b a + \epsilon}) for some constant \epsilon > 0,
#       and af(n/b) \le cf(n) for some constant c < 1 and all sufficiently large
#       n, then T(n) = \Theta(f(n)).
#
# For our recurrence, n^{log_b a} = n^{log_2 1} = 0. But f(n) = \Theta(n) =
# \Theta(n^{log_2 1 + \epsilon}), where \epsilon = 1. For sufficiently large n,
# af(n/b) = n/2 = cf(n) for c = 1/2. Consequently, by case 3, the solution to
# the recurrence is T(n) = \Theta(n).
#
# Ref: Intro. Algorithms (pages?), interview question used at Microsoft.
# ------------------------------------------------------------------------------
def missing_integer(A):
    def missing_integer_int(A, j):
        # base case -- finished searching entire array
        if len(A) == 0:
            return 0

        bit_set   = [] # entries where j-th bit was set
        bit_unset = [] # entries where j-th bit was unset
        for x in A:
            # check j-th bit
            if x & 2**j:
                bit_set.append(x)
            else:
                bit_unset.append(x)

        if len(bit_set) < len(bit_unset):
            # the missing bit is a 1
            return 2**j + missing_integer_int(bit_set, j+1)
        else:
            # the missing bit is a 0
            #
            # Note that we break ties by picking 0, this is required for the
            # algorithm to work when N is even!
            return missing_integer_int(bit_unset, j+1)

    return missing_integer_int(A, 0)

# This is a variation of the missing integer problem. You are given a sequence
# A[1..N] of N numbers that follow some arithmetic progression, with one of the
# numbers missing. Find the missing number. For example, the sequence: [1, 3, 7,
# 9, 11, 13], the missing number is 5.
#
# Our solution takes advantage of the fact that the input array is sorted to
# find the missing number in logarithmic time. We calculate the difference
# between the terms (taking advantage of the fact that we must always have the
# first and last terms in the sequence), then we check the element in the middle
# of the sequence to see if it matches what we expect. If it does, we know the
# missing integer must be in the top half of the array, so we recurse on that
# half (and vice versa if the element isn't what we expect).
def missing_integer_arithmetic_progression(A):
    print A
    d = (A[-1]-A[0])/len(A)
    if len(A) == 2:
        return A[0]+d
    else:
        p = len(A)/2
        if A[p] == A[0]+p*d:
            return missing_integer_arithmetic_progression(A[p:])
        else:
            return missing_integer_arithmetic_progression(A[0:p+1])

# -----------------------------------------------------------------------------
# Randomly return index of max value from array
# -----------------------------------------------------------------------------
#
# Question
# --------
# Given an array of integers, return the index of the maximum element. If there
# are several elements reaching the maximum, choose one uniformly at random.
#
# Solution
# --------
# As we loop over the array, we store the index of the max value that we plan to
# return. Every time we see another instance of the max value, we replace the
# saved index with the index of that instance with probability 1/x, where x is
# number of max values we have seen so far.
# -----------------------------------------------------------------------------

# This is a basic solution which requires O(N) time, and O(N) additional space
# (in the worst case).
def get_index_of_max_value_from_array_basic(A):
    max_value = A[0]
    max_indices = [0]
    for i in xrange(0, len(A)):
        if A[i] > max_value:
            max_value = A[i]
            max_indices = [i]
        elif A[i] == max_value:
            max_indices.append(i)
    return random.choice(max_indices)

# This is a slightly better solution: still O(N) time, but O(1) extra space.
def get_index_of_max_value_from_array_better(A):
    max_value = A[0]
    max_count = 1
    for i in xrange(0, len(A)):
        if A[i] > max_value:
            max_value = A[i]
            max_count = 1
        elif A[i] == max_value:
            max_count += 1
    j = random.randint(0, max_count - 1)
    max_count = 0
    for i in xrange(0, len(A)):
        if A[i] == max_value:
            if max_count == j:
                return i
            else:
                max_count += 1
    # not reached, we should always return a value above
    raise Exception('not reached')

# This version is the best posible: we do a single pass over the array, and only
# use a constant amount of additional space. The trick is to understand that we
# need to keep track of the index that we plan to return; each time we see the
# max value again, we overwrite our saved index with the current index with
# probability 1/x, where x is the number of times we've seen the max value so
# far.
def get_index_of_max_value_from_array_best(A):
    max_value = A[0]
    max_count = 1
    j = 0 # saved index to return
    for i in xrange(0, len(A)):
        if A[i] > max_value:
            max_value = A[i]
            max_count = 1
            j = i
        elif A[i] == max_value:
            max_count += 1
            if random.uniform(0, 1) < 1.0 / max_count:
                j = i # lucky winner
    return j

# This version just picks a uniformly distributed random number each time we see
# the max, and saves the index of the new instance if the new random number is
# greater than the saved value.
def get_index_of_max_value_from_array_best_alt(A):
    max_value = A[0]
    r = 0 # random number from last max
    j = 0 # saved index to return
    for i in xrange(0, len(A)):
        if A[i] > max_value:
            max_value = A[i]
            r = random.uniform(0, 1)
            j = i
        elif A[i] == max_value:
            s = random.uniform(0, 1)
            if s > r:
                r = s
                j = i # lucky winner
    return j

# ------------------------------------------------------------------------------
# Making Change
# ------------------------------------------------------------------------------
#
# Question
# --------
# Given an amount, N, find the total number of ways to make change for M, using
# pennies, nickels, dimes, and quarters.
#
# Solution
# --------
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
#
# Ref: SICP (pages?)
# ------------------------------------------------------------------------------
@memoized
@countcalls
def make_change(n, d):
    if n == 0:
        return 1
    elif n < 0 or len(d) == 0:
        return 0
    else:
        return make_change(n - d[-1], d) + make_change(n, d[:-1])

# ------------------------------------------------------------------------------
# Adding Up
# ------------------------------------------------------------------------------
#
# Like making change, but we count the ways to sum the numbers (1, N-1) to make
# N.
#
# Interestingly, these numbers are the partition numbers (Sloane's A000041),
# with the only difference being that we don't count 0 + N as a partition of
# N.
#
# See also http://mathworld.wolfram.com/PartitionFunctionP.html.
#
# Ref: interview question used at Facebook.
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Counting Palindromes
# ------------------------------------------------------------------------------
#
# Question
# --------
# Given a string, S, count all of the palindromes in the string. For example,
# the string 'abba' contains the following palindromes: 'a', 'b', 'b', 'a',
# 'bb', and 'abba', for a total of 6.
#
# Solution
# --------
# This is an O(n^2) solution that counts the number of palindromes at each index
# of the string. I believe that we can do better, however -- we should be able
# to do this in linear time...
#
# Ref: interview question used at Facebook.
# ------------------------------------------------------------------------------
def count_palindromes(s):
    def count_odd_palindromes(s, i):
        c = 1 # trivial palindrome
        d = 1 # distance to check
        while i - d >= 0 and i + d < len(s) and s[i-d] == s[i+d]:
            print 'found odd palindrome {0} at i = {1}, d = {2}'.format(s[i-d:i+d+1], i, d)
            c += 1
            d += 1
        return c

    def count_even_palindromes(s, i):
        c = 0
        d = 1
        while i - (d - 1) >= 0 and i + d < len(s) and s[i-(d-1)] == s[i+d]:
            print 'found even palindrome {0} at i = {1}, d = {2}'.format(s[i-(d-1):i+d+1], i, d)
            c += 1
            d += 1
        return c

    c = 0
    for i in range(0, len(s)):
        print 'checking index {0}'.format(i)
        c += count_odd_palindromes(s, i)
        c += count_even_palindromes(s, i)
    return c

# ------------------------------------------------------------------------------
# Maximum Sub-Array and Variations
# ------------------------------------------------------------------------------
#
# Question
# --------
# Given an array of pairs of the form (a, b), find a sub-array s.t. the 1st
# element of each pair is in increasing order and the sum of the second element
# of each pair is the maximum possible.
#
# Solution
# --------
# Kadane's algorithm can be used to find the maximum sub-array, so let's start
# with that. Then we modify it s.t. we reset the maximum sum seen so far each
# time we see an inversion in the first element in the pair.
#
# Ref: interview question used at Google.
# ------------------------------------------------------------------------------
def maximum_subarray(A):
    max_ending_here = 0
    max_so_far = 0
    for x in A:
        max_ending_here = max(0, max_ending_here+x)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

# This variant of Kadane's algorithm also handles the case where the array
# contains only negative numbers (in which case, the maximum contiguous sum is
# simply the largest number in the array).
def maximum_subarray_v2(A):
    max_ending_here = A[0]
    max_so_far = A[0]
    for x in A:
        print 'max_ending_here: {0}, max_so_far: {1}'.format(max_ending_here, max_so_far)
        max_ending_here = max(x, max_ending_here+x)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far;

def maximum_subarray_in_order(A):
    max_ending_here = max_so_far = A[0][1]
    max_so_far_start = max_so_far_end = 0
    start = end = 0
    for i in range(1, len(A)):
        if A[i][0] <= A[i-1][0]:
            # the first element in this pair isn't in increasing order, so
            # reset the sum
            max_ending_here = A[i][1]
            start = end = i
        else:
            # keep going...
            max_ending_here = max_ending_here+A[i][1]
            end = i

        if max_ending_here > max_so_far:
            max_so_far_start = start
            max_so_far_end = end
            max_so_far = max_ending_here
    print 'Max sequence was {0} with sum {1}.'.format(A[max_so_far_start:max_so_far_end+1], max_so_far)
    return A[max_so_far_start:max_so_far_end+1]

# ------------------------------------------------------------------------------
# Phone Numbers
# ------------------------------------------------------------------------------
#
# Question
# --------
# Given a phone number, for example 555-1212, print out all possible strings
# that can be made from that number by substituing the letters found on the
# keypad of a telephone. For example, 623-6537 => 'MCDNLDS' (among others).
#
# Solution
# -------
# We simply use a tree recursive process. For each position in the number, we
# try each of the possible letters in turn, recursing on the remaining digits
# each time. We we run out of digits, we print the string.
#
# Ref: interview question used at Nvidia.
# ------------------------------------------------------------------------------
def phone_numbers(D):
    digits = { 0: '0', 1: '1', 2: '2ABC', 3: '3DEF', 4: '4GHI', 5: '5JKL',
               6: '6MNO', 7: '7PQRS', 8: '8TUV', 9: '9WXYZ' }

    def phone_numbers_int(D, s):
        if len(D) == 0:
            print s
        else:
            for d in digits[D[0]]:
                phone_numbers_int(D[1:], s+d)

    phone_numbers_int(D, '')

# ------------------------------------------------------------------------------
# String Reversal
# ------------------------------------------------------------------------------
#
# Question
# --------
# Write a function to reverse the order of the words in a string in place.
#
# Solution
# --------
# Simply go through the string, swapping the first character with the last, then
# the second with the second-to-last, etc. Then, go through the string again,
# looking for spaces. For each word found this way, reverse the characters in
# the word.
#
# This solution doesn't handle punctutation.
#
# Ref: interview question used at Google.
# ------------------------------------------------------------------------------
def reverse_string(S):
    S = list(S)

    def swap(i,j):
        x = S[i]
        S[i] = S[j]
        S[j] = x

    for i in range(0, len(S)/2):
        swap(i, len(S)-(i+1))

    i = 0
    while i < len(S):
        j = i+1
        while j < len(S) and S[j] != ' ':
            j += 1
        k = j-1
        while i < k:
            swap(i,k)
            i += 1
            k -= 1
        i = j+1

    return ''.join(S)

# ------------------------------------------------------------------------------
# Order Statistics
# ------------------------------------------------------------------------------
#
# Question
# --------
# Given an unordered array of numbers, find the i-th largest, where the 0-th
# largest is the smallest number in the array.
#
# A variation of this asks about minimizing the sum of the perpendicular
# distances to a given line (for example, choosing the location of a new
# street given the (x,y) positions of a set of houses). The solution is to
# find the median distance to the x axis; putting the road there minimizes
# the overall distance from all of the houses.
#
# Solution
# --------
# We use the partition algorithm from quicksort with a random pivot. This gives
# us a solution that runs in linear time on average. We can improve this to be
# linear in the worst case if we use a 'median-of-medians' approach to select
# the pivot -- this code doesn't do that, however.
#
# Ref: interview question used at Facebook.
# ------------------------------------------------------------------------------
def find_ith_largest_number(A,i):
    def swap(A,i,j):
        print 'swapped A[{0}]={1} and A[{2}]={3}'.format(i,A[i],j,A[j])
        x = A[i]
        A[i] = A[j]
        A[j] = x

    def partition(A,p,r):
        x = A[p]
        i = p-1
        j = r+1
        while True:
            j -= 1
            while A[j] > x:
                j -= 1
            i += 1
            while A[i] < x:
                i += 1
            if i < j:
                swap(A,i,j)
            else:
                return j

    def randomized_partition(A,p,r):
        i = random.randint(p,r)
        print 'pivot is A[{0}] = {1}'.format(i,A[i])
        swap(A,i,p)
        return partition(A,p,r)

    def select(A,p,r,i):
        print 'select({0}, {1}, {2}, {3})'.format(A,p,r,i)
        print A
        if p == r:
            return A[p]
        q = randomized_partition(A,p,r)
        k = q-p+1 # number of elements in lower half
        print 'q =', q, ', k =', k
        if i < k:
            return select(A,p,q,i)
        else:
            return select(A,q+1,r,i-k)

    return select(A,0,len(A)-1,i)

# ------------------------------------------------------------------------------
# Graph Problems
# ------------------------------------------------------------------------------
#
# Question
# --------
# Given an undirected graph G = (V, E), print all possible paths from a given
# source vertex a to a given target vertex b.
#
# A variant is to ask for the total number of distinct vertices visited on all
# simple paths from a -> b; this is trivally done by adding each vertex on every
# path we discover to a set, then counting the keys in the set once we've
# finished.
#
# Solution
# --------
# We use a variant of depth-first search.
#
# Ref: interview question used at Facebook.
# ------------------------------------------------------------------------------
def find_all_paths(G,a,b):
    Q = {} # set of visited nodes

    def find_path(G,a,b,P):
        if a == b:
            print P+[b]
            return

        # color this node to prevent us from entering it again
        Q[a] = 1

        # recursively try each edge incident on a
        for v in G[a]:
            if not v in Q:
                find_path(G,v,b,P+[a])

        del Q[a]

    find_path(G,a,b,[])
