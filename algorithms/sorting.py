# Sorting algorithms!

import math
import random
import time

# First, some preliminaries...

# ------------------------------------------------------------------------------
# uniform_integer_sequence
#
# Returns a generator that yields an unbounded sequence of uniformly distributed
# random integers over the range [a, b).
# ------------------------------------------------------------------------------
def uniform_integer_sequence(a, b):
    while True:
        yield int(random.uniform(a, b))

def uniform_array(N, a, b):
    g = uniform_integer_sequence(a, b)
    return [g.next() for i in range(0, N)]

# ------------------------------------------------------------------------------
# elapsed_time
#
# Function that measures the elapsed time taken by some f(*args). Note that this
# is likely to be much less precise than the available python profilers.
# ------------------------------------------------------------------------------
def elapsed_time(f, *args):
    t1 = time.clock()
    f(*args)
    return time.clock() - t1

# Sorting algorithms

# ------------------------------------------------------------------------------
# Insertion sort
# ------------------------------------------------------------------------------
#
# Insertion sort has a worst-case running time of $$O(n^2)$$. This occurs when
# the input sequence is already in reverse sorted order (i.e., decreasing order
# for this implementation). In the best case, where the input is already sorted,
# insertion sort runs in $$O(n)$$ time.
#
# Note that, for small enough values of n, the constant factors hidden in the
# $$\Theta(n\,lg\,n)$$ running time of merge sort may make it slower than
# insertion sort in practice. For this reason, some implementations of merge
# sort apply insertion sort directly to the input array once its size becomes
# smaller than some threshold, essentially cutting off the lowest levels of the
# recursion tree.
#
# Insertion sort sorts the array in place, using only a constant amount of
# additional space.
#
# ------------------------------------------------------------------------------
def insertion_sort(A):
    for j in range(1, len(A)):
        key = A[j]
        i = j - 1
        while i >= 0 and A[i] > key:
            A[i + 1] = A[i]
            i = i - 1
        A[i + 1] = key

# --------------------------------------------------------------------------------
# Merge sort
#
# Merge sort is a recursive sorting algorithm that divides the input array into
# two halves at each step. The two halves are then recursively split into arrays
# of n/4 elements, n/8, etc., until at the lowest level of the recursion tree,
# we are left with one element in each array. These one element arrays are
# already sorted by definition. There are log_2(n) + 1 levels in the tree,
# assuming that n is a power of two.
#
# After the recursive calls, merge sort then merges the two sorted arrays of
# length $$k = (1/2^i)*n$$ into a single sorted array of length $$2k$$. The
# merge procedure works by repeatedly examining the smallest element in each
# sorted array, and moving the smaller of the two into an output array. Since
# the merge procedure runs in time $$\Theta(n)$$, the running time of the
# algorithm as a whole is defined by the recurrence relation:
#
#   $$T(n) = 2T(n/2) + \Theta(n)$$
#
# Which runs in time $$\Theta(n\,lg\,n) overall, by the master method.
#
# Note that merge sort does not sort in place! The space complexity of merge
# sort is left as an exercise to the reader.
# --------------------------------------------------------------------------------
def merge_sort(A):
    def merge(A, B):
        C = []
        i = 0
        j = 0
        while i < len(A) and j < len(B):
            if A[i] < B[j]:
                C.append(A[i])
                i = i+1
            else:
                C.append(B[j])
                j = j+1

        while i < len(A):
            C.append(A[i])
            i = i+1

        while j < len(B):
            C.append(B[j])
            j = j+1

        return C

    if len(A) <= 1:
        return A
    n = int(math.floor(len(A)/2))
    B = merge_sort(A[0:n])
    C = merge_sort(A[n:])
    return merge(B, C)

# ------------------------------------------------------------------------------
# Heapsort
#
# Heapsort's running time is O(n lg n). It sorts an aray in place, using only a
# constant amount of additional space.
#
# Heapsort uses a data structure, called a heap, to manage data during the
# execution of the algorithm. Heaps also make efficient priority queues!
# ------------------------------------------------------------------------------
def heapsort(A):
    def heap_parent(i):
        return math.floor(i / 2)
    pass

# --------------------------------------------------------------------------------
# Even-Odd Sort
#
# Partitions an array s.t. all even numbers come before all odd numbers. This
# can be done using a single pass over the array, in linear time, using a
# constant amount of additional space.
#
# This is simply a special case of the partition function used by quicksort.
#
# Ref: interview question used at VMware.
# --------------------------------------------------------------------------------
def even_odd_sort(A):
    i = 0
    j = len(A) - 1
    while i < j:
        print 'i = {0}, j = {1}.'.format(i, j)
        while A[i] % 2 == 0 and i <= j:
            print 'Moved i to {0}.'.format(i + 1)
            i += 1
        while A[j] % 2 == 1 and j >= i:
            print 'Moved j to {0}.'.format(j - 1)
            j -= 1
        if i != j and i < j:
            print 'Swapping A[{0}] and A[{1}].'.format(i, j)
            x = A[i]
            A[i] = A[j]
            A[j] = x
