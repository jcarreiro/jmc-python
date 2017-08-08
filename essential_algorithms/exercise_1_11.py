#!/usr/bin/python

import math
import numpy as np
import matplotlib.pyplot as plt

def log(t):
    return [math.log(x, 2) for x in t]

def sqrt(t):
    return [math.sqrt(x) for x in t]

def linear(t):
    return t

def squared(t):
    return [x ** 2 for x in t]

def exponential(t):
    return [2 ** x for x in t]

def factorial(t):
    return [math.factorial(x) for x in t]

def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n == 2:
        return 1
    else:
        i = 1 # fib(n-1)
        j = 1 # fib(n-2)
        for x in xrange(3, n+1):
            k = i + j # fib(n) = fib(n-1) + fib(n-2)
            j = i
            i = k
        return i

def fibonacci(t):
    return [fib(x) for x in t]

def plot_functions():
    fns = [
        ['log n', log],
        ['sqrt n', sqrt],
        ['n', linear],
        ['n^2', squared],
        ['2^n', exponential],
        ['n!', factorial],
        ['fib n', fibonacci],
    ]

    t = range(1, 101)
    for label, fn in fns:
        plt.plot(t, fn(t), label=label)
    plt.yscale('log')
    plt.legend(loc='upperleft')
    plt.show()

if __name__ == '__main__':
    plot_functions()
