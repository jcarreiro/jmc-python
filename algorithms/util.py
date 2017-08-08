import numpy as np

def pairwise_sums(A):
    # Get the pairwise sums of the elements in A.
    # A -- np.array of values
    # returns: pairwise sums of A, as an array with at most n(n-1)/2 entries
    return np.unique(np.array(np.matrix(A) + np.matrix(A).T))
