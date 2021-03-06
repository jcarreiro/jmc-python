# Numerical methods

import math

from jmc.decorators import memoized

@memoized
def fibonacci(n):
    print 'Called for n = {0}'.format(n)
    if n in (0, 1):
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def factorial(n):
    a = 1
    for b in range(2, n+1):
        a = a * b
    return a

def n_choose_k(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k));

# Multiplication using the built-in '*' operator.
def multiply(a, b):
    return a * b

# Recursive multiplication.
#
# NOTE: both of the following methods have bad numerical instability problems!
#
# Given two n-digit numbers ab and cd, note that:
#
#   ab * cd = (10^(n/2) * a + b) * (10^(n/2) * c + d)
#           = 10^n * a * c + 10^(n/2) * (a * d + b * c) + b * d
#
# So we can recusively solve the subproblems a * c, a * d, b * c, and b * d,
# on numbers with n/2 digits.
def recursive_multiply(x, y):
    # Base case
    if x < 10 or y < 10:
        return x * y

    n = max(math.ceil(math.log(x, 10)), math.ceil(math.log(y, 10)))
    a = x/pow(10, n/2)
    b = x - a * pow(10, n/2)
    c = y/pow(10, n/2)
    d = y - c * pow(10, n/2)

    print 'number of digits = {}.'.format(n)
    print 'a = {0}, b = {1}.'.format(a, b)
    print 'c = {0}, d = {1}.'.format(c, d)

    return pow(10, n) * recursive_multiply(a, c) + pow(10, n/2) * (recursive_multiply(a, d) + recursive_multiply(b, c)) + recursive_multiply(b, d)

# Karatsuba multiplication.
def karatsuba_multiply(x, y):
    if x < 10 or y < 10:
        return x * y

    n = max(math.ceil(math.log(x, 10)), math.ceil(math.log(y, 10)))
    a = x/pow(10, n/2)
    b = x - a * pow(10, n/2)
    c = y/pow(10, n/2)
    d = y - c * pow(10, n/2)

    z2 = karatsuba_multiply(a, c)
    z0 = karatsuba_multiply(b, d)
    z1 = karatsuba_multiply(a + b, c + d) - z2 - z0

    return pow(10, n) * z2 + pow(10, n/2) * z1 + z0

# Ackermann's function:
#            _
#           /
#           | y + 1                    if x = 0
# A(x,y) = <  A(x - 1, 1)              if y = 0
#           | A(x - 1, A(x, y - 1))    otherwise
#           \_
#
# (That is supposed to be a brace, above.)
#
# Some simple optimizations:
#
#   A(1,n) = n + 2
#   A(2,n) = 2n + 3
#   A(3,n) = 2^(n + 3) - 3
#
# Without these, it is not possible to compute even A(4,1) recursively in
# Python without exceeding the amount of available memory on my computer!
def ackermann(x, y):
    if x == 0:
        return y + 1
    elif x == 1:
        return y + 2
    elif x == 2:
        return 2 * y + 3
    elif x == 3:
        return pow(2, y + 3) - 3
    elif y == 0:
        return ackermann(x - 1, 1)
    else:
        return ackermann(x - 1, ackermann(x, y - 1))

# Helper function used by ackermann2.
def ack2(x, y, count, depth, max_depth):
    depth = depth + 1
    n = 0
    if x == 0:
        n = y + 1
        max_depth = max(depth, max_depth)
    elif y == 0:
        n, count, max_depth = ack2(x - 1, 1, count, depth, max_depth)
    else:
        n, count, max_depth = ack2(x, y - 1, count, depth, max_depth)
        n, count, max_depth = ack2(x - 1, n, count, depth, max_depth)
    return n, count + 1, max_depth

# This version doesn't have the optimizations above, and it also tracks the
# total number of calls to ack().
def ackermann2(x, y):
    n, count, max_depth = ack2(x, y, 0, 0, 0)
    print '{0} calls'.format(count)
    print 'Max depth was {0}'.format(max_depth)
    return n

def gcd(x, y):
    if y == 0:
        return x
    return gcd(y, x % y)

# Approximates the definite integral of f over the interval (a, b) using the
# trapezoidal method with n panels.
def trapezoidal(f, a, b, n):
    pass
