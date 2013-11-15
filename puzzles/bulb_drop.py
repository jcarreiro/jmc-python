# Bulb dropping puzzle, as seen on http://qntm.org/bulbs:
#
#   The problem
#
#   You have a 100-story building and 2 identical, supposedly-unbreakable
#   light bulbs. You know for a fact that if you drop a bulb at floor 0 (i.e.
#   ground level), the bulb will not break. You also know for a fact that if
#   you drop a bulb at floor 101 (i.e. go to the roof and throw it off), the
#   bulb will definitely break. In fact, there is some floor f with 0 <= f <=
#   100 such that if you drop a bulb at floor f or lower it will not break,
#   and if you drop it at floor f+1 or higher, it will definitely break. You
#   have only 2 light bulbs. Once they are both broken, there are no more.
#   Assume that dropping a bulb multiple times does not weaken it. How many
#   bulb drops must be performed to find f, in the worst-case scenario?
#
# The answer is a recurrence relation. For a single bulb, the recurrence
# relation is:
#
#   OneBulb(0) = 0
#   OneBulb(f) = 1 + OneBulb(f - 1)
#
# Or, OneBulb(f) = f, where f is the number of floors. This makes sense
# because with a single bulb, we are forced to check every floor, starting
# with the lowest.
#
# For 2 bulbs, our strategy changes: we can now test at least one floor, k,
# "in the middle" of the building. If the bulb breaks, then we must check each
# floor from the lowest up to k; if it does not, then we have reduced the
# problem to the set of floors above k. The worst case is whichever of these
# possibilities requires the largest number of trials, so:
#
#   TwoBulbs(0) = 0
#   TwoBulbs(f) = 1 + max(OneBulb(k - 1), TwoBulbs(f - k))
#
# where k is the floor where the initial bulb is dropped, and 1 <= k <= f. The
# "best" worst case, is whatever value of k results in the minimum value of
# that function.
#
# In the general case, where we have b bulbs, this becomes:
#
#   Drops(b, 0) = 0
#   Drops(1, f) = 1 + Drops(1, f - 1)
#   Drops(b, f) = 1 + max(Drops(b - 1, k - 1), Drops(b, f - k))
#
# i.e., we either recurse on the floors < k with one less bulb, or we recurse
# on the floors > k with all of our bulbs still intact.

def OneBulb(f):
    '''Returns the number of drops needed to solve the Bulb Drop problem for
       the case where only a single bulb is available.'''
    return f

# Cached used for memoization.
cache = {}

def TwoBulbs(f):
    '''Tree-recursive function that returns the number of drops needed to solve
       the bulb drop problem in the two bulb case. Uses memoization to avoid
       redundant calculations.'''
    try:
        return cache[f]
    except KeyError:
        answer = 0
        if f != 0:
            answer = 1 + min(map(lambda k: max(OneBulb(k-1), TwoBulbs(f-k)), range(1, f+1)))
        cache[f] = answer
        return answer

def Drops(b, f):
    '''Tree-recursive function that solve the bulb drop problem in the general
       case. Uses memoization to avoid redundant calculations.'''
    pass
