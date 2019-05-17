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
#   - largest runway on island
#   - validate BST
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections # namedtuple
import os
import random
import sys
import unittest

# Fix up import path.
#
# TODO: figure out how to do module-relative imports
sys.path.append(os.path.expanduser('~/src/python'))

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
  def helper(s, t):
    if len(s) == 0:
      print(t)
    else:
      for i in xrange(0, len(s)):
        helper(s[0:i] + s[i+1:], t + s[i])
  helper(s, '')

# More than one candidate has given me this "solution". Since it only prints
# n^2 strings and there are n! permutations, it can't possibly be correct...
def permutations_wrong(s):
    for i in xrange(len(s)):
        c = s[i]
        t = s[0:i] + s[i+1:]
        for j in xrange(len(t) + 1):
            yield t[0:j] + c + t[j:]

# Another bad solution attempt -- this one tries to print one character at a
# time, which can't work since we need to print the character at position N
# in the string (N-1)! times (once for each permutation in which it occurs at
# that position).
def permutations_wrong2(s):
    if not s:
        print() # print a newline at the end
    for i in xrange(len(s)):
        print(s[i], end='')
        permutations_wrong2(s[0:i] + s[i+1:])

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

def permutations_yield(s):
    if len(s) == 1:
        yield s
    elif len(s) == 0:
        raise ValueError()
    else:
        for i in range(len(s)):
            rest = s[0:i] + s[i+1:]
            for p in permutations_yield(rest):
                yield s[i] + p

# An elegant solution that builds the permutation in place inside the input
# string.
def permutations_swap(s):
    def swap(s, a, b):
        x = s[a]
        s[a] = s[b]
        s[b] = x

    def helper(s, i):
        if i == len(s):
            # print ''.join(s)
            return

        for j in range(i, len(s)):
            swap(s, i, j)
            helper(s, i+1)
            swap(s, i, j)
    helper(list(s), 0)

# More than one candidate has given me this solution. Initially I thought the
# complexity was worse than factorial but now I'm not so sure.
def permutations_dfs(s):
  def helper(s, p, d):
    if len(p) == len(s):
      return # print p
    else:
      for c in s:
        if not d[c]:
          d[c] = True
          helper(s, p + c, d)
          d[c] = False
  d = {}
  for c in s:
    d[c] = False
  helper(s, '', d)

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
    print(A)
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
# number of max values we have seen so far. This is also known as "reservoir
# sampling" (https://en.wikipedia.org/wiki/Reservoir_sampling).
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
# of the string.
#
# TODO: can we do this in linear time?
#
# Ref: interview question used at Facebook.
# ------------------------------------------------------------------------------
def count_palindromes(s):
    def count_odd_palindromes(s, i):
        c = 1 # trivial palindrome
        d = 1 # distance to check
        while i - d >= 0 and i + d < len(s) and s[i-d] == s[i+d]:
            print('Found odd palindrome {0} at i = {1}, d = {2}'.format(s[i-d:i+d+1], i, d))
            c += 1
            d += 1
        return c

    def count_even_palindromes(s, i):
        c = 0
        d = 1
        while i - (d - 1) >= 0 and i + d < len(s) and s[i-(d-1)] == s[i+d]:
            print('Found even palindrome {0} at i = {1}, d = {2}'.format(s[i-(d-1):i+d+1], i, d))
            c += 1
            d += 1
        return c

    c = 0
    for i in range(0, len(s)):
        print('Checking index {0}'.format(i))
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
        print('max_ending_here: {0}, max_so_far: {1}'.format(max_ending_here, max_so_far))
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
    print('Max sequence was {0} with sum {1}.'.format(A[max_so_far_start:max_so_far_end+1], max_so_far))
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
            print(s)
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
        print('Swapped A[{0}]={1} and A[{2}]={3}'.format(i,A[i],j,A[j]))
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
        print('Pivot is A[{0}] = {1}'.format(i,A[i]))
        swap(A,i,p)
        return partition(A,p,r)

    def select(A,p,r,i):
        print('Select({0}, {1}, {2}, {3})'.format(A,p,r,i))
        print(A)
        if p == r:
            return A[p]
        q = randomized_partition(A,p,r)
        k = q-p+1 # number of elements in lower half
        print('q =', q, ', k =', k)
        if i < k:
            return select(A,p,q,i)
        else:
            return select(A,q+1,r,i-k)

    return select(A,0,len(A)-1,i)

# ------------------------------------------------------------------------------
# Graph and Tree Problems
# ------------------------------------------------------------------------------

# -----------------------------------------------------------------------------
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
            print(P + [b])
            return

        # color this node to prevent us from entering it again
        Q[a] = 1

        # recursively try each edge incident on a
        for v in G[a]:
            if not v in Q:
                find_path(G,v,b,P+[a])

        del Q[a]

    find_path(G,a,b,[])

# ------------------------------------------------------------------------------
# Question: Given a binary tree (but not necessarily a binary search tree), find
# the length of the shortest path between two nodes.
#
# Solution: We perform a depth-first search, returning from each recursive call
# a state vector where the first element is:
#   - 0 if neither node has been found;
#   - 1 if one node has been found, and;
#   - 2 otherwise;
#
# and the second element is the current path length, which is the sum of the
# path lengths for the left and right subtrees, plus 1 if we have seen one of
# the nodes we're searching for, but not both.
#
# This solution assumes that nodes in the tree are unique.
#
# Reference: Interview question used at Facebook.
# ------------------------------------------------------------------------------
def length_of_shortest_path_in_tree(T, a, b):
    def helper(T, a, b):
        if len(T) == 0:
            return (0, 0)
        l = helper(T[1], a, b)
        r = helper(T[2], a, b)
        s = l[0] + r[0]
        if T[0] == a or T[0] == b:
            s += 1
        n = l[1] + r[1]
        if s == 1:
            n += 1
        return (s, n)
    return helper(T, a, b)[1]

# Class used to represent a node in a binary tree.
BNode = collections.namedtuple('BNode', ['value', 'left', 'right'])

# ------------------------------------------------------------------------------
# Question: Given a binary tree, T, find the value of the maximum path in T.
#
# Solution: We define a path as a sequence of nodes (a, b, ..., g) s.t. each
# node in the sequence has an edge connecting it to the previous and the next
# node in the sequence. We define the value of a path as the sum of the values
# of the nodes in the sequence, v(p) = \sum n forall n in p.
#
# This problem admits an easy recursive solution. Consider an arbitrary node
# n in the tree. Then there are three possible cases we need to consider:
#
#   1. The max path is contained within the node's left subtree.
#   2. The max path goes through this node.
#   3. The max path is contained within the node's right subtree.
# ------------------------------------------------------------------------------
def max_path_in_tree(t):
    def helper(t):
        if t is None:
            return (0, 0)
        l_contained, l_through = helper(t.left)
        r_contained, r_through = helper(t.right)
        # Our max is one of:
        #   1. the best path contained in our left subtree
        #   2. the path through us that continues into our left subtree
        #   3. the path through us that continues into our right subtree
        #   4. the best path contained in our right subtree
        #   5. the best path contained in our left subtree, through us,
        #      into our right subtree
        # (whichever is greater).
        max_contained = max(l_contained,
                            r_contained,
                            l_through + r_through + t.value)
        max_through = max(l_through + t.value, r_through + t.value)
        return (max_contained, max_through)
    a, b = helper(t)
    return max(a, b)

# -----------------------------------------------------------------------------
# Word Walk
#
# Given a starting word, an ending word, and a dictionary, return a list of
# valid words which transforms the start word into the end word. Successive
# words in the list can differ by at most one character. All words on the list
# must be in the provided dictionary.
#
# For example, given the start "Walk" and the end "Bard", and assuming the
# dictionary is a typical dictionary of English words, one possible result
# could be: ["walk", "balk", "bald", "bard"].
#
# Solution:
#
# Ref: Interview question used at Facebook (and I swear I saw this in an ad
# on the T one time...).
# -----------------------------------------------------------------------------
def word_walk(start, end, dictionary):
    pass

# -----------------------------------------------------------------------------
# Array and List Problems
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Question: Determine if any three integers in an array sum to 0.
#
# Solution: We assume that reptitions are allowed. The naive solution is simply
# to check all possible combinations, one at a time. This has O(n^3) complexity.
#
# We can do slightly better by first inserting all of the values in the array
# into a hash table; this optimizes our inner loop, reducing the complexity to
# O(n^2) but requiring O(n) space for the hash table.
#
# Finally we can maintain the O(n^2) complexity but reduce the extra space
# required to O(1) by sorting the input list, then, for each element in the
# list, performing a linear search over the sorted list to find the pair of
# elements (if any) for which the sum of all three elements is zero.
#
# Ref: Interview question used at Facebook.
#-------------------------------------------------------------------------------
def list_three_int_sum_to_zero_naive(A):
    solutions = []
    for i in range(len(A)):
        for j in range(i, len(A)):
            for k in range(j, len(A)):
                if A[i] + A[j] + A[k] == 0:
                    solutions.append((i, j, k))
    return solutions

# This solution uses hash table to slightly improve the complexity (to O(n^2))
# but requires O(n) extra space.
def list_three_int_sum_to_zero_hashtable(A):
    solutions = []
    d = {}
    for i in range(len(A)):
        d[A[i]] = i
    for i in range(len(A)):
        for j in range(i, len(A)):
            sum = -(A[i] + A[j])
            if sum in d:
                solutions.append((i, j, d[sum]))
    return solutions

# This solution first sorts the list, then uses a linear search over the right-
# hand side of the list for each element in it. This maintains the quadratic
# complexity, but requires no extra space.
def list_three_int_sum_to_zero(A):
    solutions = []
    A.sort()
    for i in range(len(A) - 2):
        j = i + 1
        k = len(A) - 1
        while j <= k:
            sum = A[i] + A[j] + A[k]
            if sum == 0:
                solutions.append((i, j, k))
                j += 1
                k -= 1
            elif sum < 0:
                # We need a larger number; move the left end of the range up
                j += 1
            else:
                # We need a smaller number; move the right end of the range down
                k -= 1
    # There are some special cases to handle. Any 0s in the array are also a
    # solution (simply pick 0 three times). Finally, if we find A[j] = -2 * A[i]
    # in the array for some some i, j, then (i, i, j) is also a solution.
    #
    # Note that this additional quadratic step doesn't change our asymptotic
    # complexity.
    for i in range(len(A)):
        if A[i] == 0:
            solutions.append((i, i, i))
        else:
            for j in range(i + 1, len(A)):
                if 2 * A[i] + A[j] == 0:
                    solutions.append((i, i, j))

    return solutions

# ------------------------------------------------------------------------------
# Island Airport
#
# Question: Consider a grid where each cell may either be water or land. The
# non-water cells make up a set of "islands" (note that there may be more than
# one disconnected island). You would like to build an airport on one of the
# islands, subject to two constraints:
#
#   1. The airport must have two runways, one aligned N/S, the other E/W. They
#      must intersect at exactly one point (forming a cross).
#
#   2. You want the runways to have the largest area possible.
#
# Return the (x, y) coordinates of the intersection.
#
# Solution:
#
# ------------------------------------------------------------------------------

# G is the grid, as a list of lists, where G[i][j] is the cell at (i, j). The
# value at each cell is 1 if the cell is "land" and 0 otherwise.
#
# This is the most naive soution -- just try to grow the runways from every
# cell and remember the max. If the grid has length N on each side, then this
# solution is O(N^3), since for each of the N^2 grid intersections, we examine
# (in the worst case) 2N cells (N cells along each axis).
def island_runway(G):
    def grow(G, i, j, d):
        # grow the runway starting at (i,j) as far as possible in direction d;
        # then return the length in that direction. d should be a tuple of the
        # form (dy, dx).
        y = i
        x = j
        while y >= 0 and y <= i and x >= 0 and x <= j and G[y][x] == 1:
            y += d[0]
            x += d[1]
        # XXX this assumes we only ever move along one axis
        return abs(y - i) + abs(x - j)

    p = None
    max_length = 0
    for i in range(0, len(G)):
        for j in range(0, len(G[i])):
            print('Considering ({i}, {j})...'.format(i=i, j=j))
            if G[i][j] != 1:
                # this cell isn't even land
                continue
            length  = grow(G, i, j, ( 0, -1)) # go west
            length += grow(G, i, j, ( 0,  1)) # go east
            length += grow(G, i, j, ( 1,  0)) # go north
            length += grow(G, i, j, (-1,  0)) # go south
            if length > max_length:
                # found a new longest runway
                p = (i, j)
                max_length = length
    return p

# The solution above repeatedly recomputes the longest runway we can build for
# each grid row and column, so if we could avoid doing that, we could solve the
# problem in less time. One insight that might help is to realize that, even
# for rows (columns) which aren't fully connected, we can compute the maximum
# connected length for each row (column); that's the longest that the runway can
# possibly be in that dimension if we pick that intersection! Then we look for
# the maximum combined extent in the obvious way. I think?
def island_runway_fast(G):
#     row_lengths = []
#     for i in range(0, len(G)):
#         max_l = 0 # max length for this row
#         l = 0     # current length for this row
#         for j in range(0, len(G[i])):
#             if G[i][j] == 1:
#                 l += 1
#             else:
#                 l = 0 # reset when we hit water
#             if l > max_l:
#                 max_l = l
#
    pass

def display_island(G):
    key = {0: '~', 1: '_'}
    for r in G:
        for c in r:
            print('{0} '.format(key[c]), end='')
        print()

def expect(expected, actual):
    if expected != actual:
        raise ValueError('Expected {0} but got {1}!'.format(expected, actual))

def test_runway():
    # The simplest possible island is no island at all.
    G = [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    p = island_runway(G)
    print('No land:')
    display_island(G)
    expect(None, p)
    print('Runway at: None.')

    # If only one row and one column are land, then there's only one possible
    # intersection.
    G = [[0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]]
    print('Cross-shaped island:')
    p = island_runway(G)
    display_island(G)
    print('Runway at ({0}, {1}).'.format(p[1], p[0])) # put x first in output
    expect((2, 2), p)

    # Two disconnected islands, where the longest extent in x is on one and the
    # longest extent in y is on the other. Tricky!
    G = [[1, 1, 1, 1, 0],
         [1, 1, 1, 0, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 1, 1, 1]]
    print('Tricky island:')
    p = island_runway(G)
    display_island(G)
    print('Runway at ({0}, {1}).'.format(p[1], p[0])) # put x first in output
    expect((4, 4), p)

# ------------------------------------------------------------------------------
# Card Shuffling
#
# Question: A deck of 2N cards is split into two piles of exactly N cards each
# and shuffled. Generate all possible arrangements of the deck after 1, 2, ...,
# S shuffles. Note that the shuffles are not perfect shuffles!
# ------------------------------------------------------------------------------
def card_shuffle(n, s):
    pass

# ------------------------------------------------------------------------------
# McNugget Numbers
# ------------------------------------------------------------------------------

# Simple recursive solution; basically DFS on a tree, where the nodes are
# numbers and the edges represent buying carton with one of the given sizes.
def mcnugget(n, sizes=[3, 6, 20]):
    if n == 0:
        return True
    elif n < 0:
        return False
    else:
        for s in sizes:
            if mcnugget(n - s, sizes):
                return True
        return False

# Variant of the recursive solution that prints the path taken in the success case.
def mcnugget_again(n, p, sizes=[3, 6, 20]):
    if n == 0:
        print(', '.join([str(x) for x in p]))
        return True
    elif n < 0:
        return False
    else:
        for s in sizes:
            if mcnugget_again(n - s, p + [s], sizes):
                return True
        return False

# Builds the table for the DP solution and returns it.
def mcnugget_table(n, sizes=[3, 6, 20]):
    # Build a table, starting from 0. For the k-th entry in the table, we store
    # True if any of the {k-m for m \in sizes} slots are True, and False
    # otherwise.
    table = [False] * (n+1)
    table[0] = True # we can always buy 0 McNuggets
    for k in range(1, n+1):
        y = False
        for m in sizes:
            if k - m >= 0 and table[k - m]:
                y = True
        table[k] = y
    return table

def mcnugget_dp(n, sizes=[3, 6, 20]):
    table = mcnugget_table(n, sizes)
    return table[-1]

# Simple solution: we can buy N McNuggets if (given sizes of [3,6,20]),
# n % 20 == 0 or n % 3 == 0 or (n % 20) % 3 == 0.
#
# Note that this solution doesn't actually work! This is an example of a
# frequently encountered incorrect solution.
def mcnugget_simple(n, sizes=[3, 6, 20]):
    for s in sizes[::-1]:
        n = n % s
    return n == 0

# Variation of above where we remove sizes if we get a false result; still
# doesn't work.
#
# TODO: counter-example and general proof of why this won't work.
def mcnugget_simple_2(n, sizes=[3, 6, 20]):
    s = sizes[:] # copy list so pop doesn't mutate original
    while s:
        if mcnugget_simple(n, s):
            return True
        s.pop() # didn't work, try again without largest size
    return False

# Test another McNugget solution against the known good DP solution and return
# inputs where it fails.
def mcnugget_test(fn, sizes=[3, 6, 20]):
    a = set(filter(lambda x: mcnugget_dp(x, sizes), range(1, 101)))
    b = set(filter(lambda x: fn(x, sizes), range(1, 101)))
    return a ^ b

# TODO: balance a mobile with unequal weights (David's question)

# ------------------------------------------------------------------------------
# Tree Print by Levels
# ------------------------------------------------------------------------------

class Node(object):
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def __repr__(self):
        s = 'Node({0}'.format(self.data)
        if self.left:
            s += ', '
            s += repr(self.left)
        if self.right:
            s += ', '
            s += repr(self.right)
        s += ')'
        return s

def tree_print_by_levels(T):
    def helper(node, levels, depth):
        if node:
            l = levels.get(depth, [])
            l.append(node)
            levels[depth] = l
            helper(node.left, levels, depth + 1)
            helper(node.right, levels, depth + 1)


    # dictionary, keys = depth in tree, values = list of nodes at that depth
    levels = {}
    helper(tree, levels, 0)

    # print dictionary
    for k, v in levels.iteritems():
        print('{0}: {1}'.format(k, ', '.join([str(x.data) for x in v])))

# ------------------------------------------------------------------------------
# Balance Parentheses
# ------------------------------------------------------------------------------
#
# Question
# --------
# Given a string, remove the minimum number of characters necessary to balance
# the parentheses in the string. For example:
#
#   f("")        => ""
#   f("()")      => "()"
#   f(")a(b)c(") => "a(b)c"
#
# Solution
# --------
# We can do this in two passes: one to count the number of closing braces, the
# other to find the matching open braces and to do the mutations. Note that
# strings are immutable in Python; by requiring the input be a list, we can do
# the mutations in-place, so that we use only a constant amount of extra space.
# If we can't require the input to be a list then this solution requires a
# linear amount of space (we'd need to turn the string into a list ourselves so
# that we can mutate it). The resulting solution requires O(n) time.
#
# Ref: interview question used at Facebook.
# ------------------------------------------------------------------------------

# For brevity, we take a list, since they are mutable and strings are not.
#
# This solution is the least efficient. We simply do a brute force search for
# each open brace, looking for a matching close brace.
def balance_parens_bad(s):
    pass

# This solution is optimal: it uses linear time and constant additional space.
def balance_parens(s):
    remaining_close = s.count(')') # pass one, count right braces
    nest_count = 0
    j = 0
    for i in range(len(s)): # pass two, remove unmatched braces
        c = s[i]
        if c == ')':
            remaining_close -= 1
            if nest_count:
                # this close brace matches an open brace
                nest_count -= 1
            else:
                # this close brace doesn't match an open brace
                continue
        if s[i] == '(':
            if nest_count == remaining_close: # no more close braces to match
                continue
            nest_count += 1
        # if we got here, we want to output this character
        s[j] = c
        j += 1
    return s[:j] # trim list to only output characters

# -----------------------------------------------------------------------------
# List Problems
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Deep Copy List
#
# Question: Given a list of nodes, where each node has a pointer to the next
# node, and a pointer to another node on the list, return a deep copy of the
# list. The "extra" pointer in each node may point to any entry in the list,
# including the node itself, or null.
#
# Solution: We proceed from the head, copying nodes one at a time. When we
# encounter a node, there are 4 possible values for its "extra" pointer:
#
#   1) null
#   2) itself
#   3) a node we've already copied
#   4) a node we haven't encountered yet
#
# Cases (1) and (2) are trivial. To handle (3) and (4), we keep a map from
# nodes in the source list to nodes in the destination list. If the node is
# already present in the map, then we can use the value to set the "extra"
# pointer. If the node is not present yet, we allocate it and add it to the
# map.
#
# Ref: Interview question used at Facebook.
# -----------------------------------------------------------------------------

class ExtraNode(object):
    def __init__(self, data, next_=None, extra=None):
        self.data = data
        self.next_ = next_
        self.extra = extra

def print_extra_list(head):
    while head:
        extra_data = None
        if head.extra:
            extra_data = head.extra.data
        print('{0} -> {1}'.format(head.data, extra_data))
        head = head.next_

def deep_copy_list(head):
    d = {}
    prev = None
    l = None
    while head:
        # Copy the node
        n = None
        if not head in d:
            n = ExtraNode(head.data, None, None)
            d[head] = n
        else:
            n = d[head]
        print('n={}'.format(n))

        # Save a pointer to the head of the copied list.
        if not l:
            l = n

        # If we have a previous node, make it point to this one.
        if prev:
            prev.next_ = n
        prev = n

        # Fix up extra pointer.
        ex = head.extra
        if ex:
            if ex in d:
                n.extra = ex
            else:
                n.extra = ExtraNode(ex.data, None, None)
                d[ex] = n.extra

        head = head.next_

    return l

# ------------------------------------------------------------------------------
# Has Duplicates
# ------------------------------------------------------------------------------
#
# Question
# --------
# Given a list of integers of length N, in which every number is between 1 and
# N, return True iff the list contains at least one duplicate entry.
#
# Solution
# --------
# Trivial.
#
# Ref: practice interview question used at Facebook.
# ------------------------------------------------------------------------------

# Linear time, linear space.
def has_duplicates(x):
    seen = [False] * (len(x) + 1)
    for i in x:
        if seen[i]:
            return True
        else:
            seen[i] = True
    return False

# Sorting first lets us do this in-place, if we're allowed to modify the original
# list. This takes O(n ln n) time.
def has_duplicates_sort(x):
    x.sort()
    for i in range(0, len(x) - 1):
        if x[i] == x[i+1]:
            return True
    return False
