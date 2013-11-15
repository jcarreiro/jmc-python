# Search algorithms

# ------------------------------------------------------------------------------
# The Knuth-Morris-Pratt algorithm for string searching.
#
# The KMP algorithm searches an input string for all occurrences of a pattern
# in linear time. This is accomplished by precomputing a table which contains
# the width of the widest border of each prefix of the pattern, where a border
# is a proper prefix of the pattern that is also a proper suffix. For example,
# in the pattern 'ABABAA', 'A' is a border of the prefix with length 3, since
# 'A' is both a proper prefix and proper suffix of 'ABA'. Note that the prefix
# may overlap with the suffix. For example, 'ABA' is a border of the prefix
# 'ABABA'.
#
# An example of a table for the search pattern 'ABABAA' is:
#
#    j:  0 1 2 3 4 5 6
# p[j]:  a b a b a a
# b[j]: -1 0 0 1 2 3 1
#
# The precomputed table is then used to determine the distance to shift the
# search pattern when a mismatch is encountered during the search. For example,
# consider searching the string 'ABABABAA' using the pattern 'ABABAA'. After we
# have matched 5 characters of the pattern ('ABABA'), we find a mismatch. But,
# we have already matched the prefix 'ABA' of the pattern a second time. So by
# shifting the pattern forward two characters, we can continue searching without
# needing to reconsider any characters we've already searched.
#
# An example of the algorithm matching the pattern 'ABABAA' in the string
# 'ABABABAA' is:
#
# a b a b a b a a
# a b a b a x      <-- matched 5 symbols, then failed at position 5
#     a b a b a a  <-- b[5] = 3. 5 - 3 = 2, so shift forward two and continue
#
# Again, this works because after we've matched 5 characters of the pattern,
# in this specific example, the widest border contained in those 5 characters
# is 3 characters long. That means that the 3 character long suffix of the
# characters we have already matched is also a prefix of the search pattern.
# So, we can keep three of the characters we have already matched and try again
# from position 3 in the search pattern.
#
# Why is this O(n)? Note that, for all j, b[j] < j by definition, so the inner
# loop of the table building algorithm always decreases j. Since the inner loop
# terminates when j = -1, it can decrease j at most as often as it has been
# incremented in the outer loop. Since the outer loop executes exactly m times,
# where m is the length of the pattern, the limit to how many times the inner
# loop can execute is O(m). The argument for the search algorithm is similar.
# ------------------------------------------------------------------------------
def kmp_search(text, pattern):
    def kmp_table(pattern):
        i = 0
        j = -1
        b = [-1 for x in range(0, len(pattern) + 1)]

        # b[0] is always -1, since the empty string has no border. This is used
        # to break the inner loop in kmp_search if we fall back all the way to
        # position 0 in the table.
        b[0] = -1
        while i < len(pattern):
            while j >= 0 and pattern[i] != pattern[j]:
                j = b[j]
            i += 1
            j += 1
            b[i] = j
        return b

    M = [] # the list of offsets where we found a match
    b = kmp_table(pattern)
    i = 0
    j = 0
    while i < len(text):
        while j >= 0 and text[i] != pattern[j]:
            # Fall back to widest border for this many matched characters.
            j = b[j]

        i += 1
        j += 1
        if j == len(pattern):
            print 'Found match at position {0}'.format(i - j)
            M.append(i - j)
            j = b[j] # shift pattern forward

# ------------------------------------------------------------------------------
# The Rabin-Karp algorithm for string searching.
#
# The Rabin-Karp algorithm searches an input string S for a substring P in
# linear time.
# ------------------------------------------------------------------------------
def rabin_karp():
    pass
