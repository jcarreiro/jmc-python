##############################################################################
# Basic compression algorithms
##############################################################################

# Prefix Codes
#
#   1. Every message is a leaf in a binary tree.
#
#   2. The code is encoded in the path from the root to the leaf (left = 0,
#      right = 1)
#
# As a result, no code is a prefix of another code.
#
# The average length of a code, assuming a probability distribution on the
# symbols, is l = \sum_{i} p_{i} l_{i}, where p_i is the probability of the
# i-th symbol, and l_i is the length of its code (the depth of the leaf).
#
# Source: Algorithms in the Real World lecture notes.

def prefix_code(s):
    # build a tree with len(s) leaf nodes

    # now just assign symbols from s to the leaf nodes until we run out

    pass
