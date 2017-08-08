# simple.py --
#
# Python implementation of the SIMPLE language from _Understanding Computation_
# by Tom Stuart.
#
# Note that printing the guillemet characters is troublesome when running Python
# from within Emacs; rather than try to debug the problem, I've substituted '<<'
# and '>>' for them below.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

class Number(object):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return '<<{}>>'.format(self)

    def reducible(self):
        return False

class Add(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return '{} + {}'.format(self.left, self.right)

    def __repr__(self):
        return '<<{}>>'.format(self)

    def reducible(self):
        return True

    def reduce(self):
        if self.left.reducible():
            return Add(self.left.reduce(), self.right)
        elif self.right.reducible():
            return Add(self.left, self.right.reduce())
        else:
            return Number(self.left.value + self.right.value)

class Multiply(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return '{} * {}'.format(self.left, self.right)

    def __repr__(self):
        return '<<{}>>'.format(self)

    def reducible(self):
        return True

    def reduce(self):
        if self.left.reducible():
            return Multiply(self.left.reduce(), self.right)
        elif self.right.reducible():
            return Multiply(self.left, self.right.reduce())
        else:
            return Number(self.left.value * self.right.value)

# This version of the Machine class is the first one we create in the book. It's
# only able to reduce expressions, and the only state it maintains is the
# current expression.
class Machine(object):
    def __init__(self, expression):
        self.expression = expression

    def step(self):
        self.expression = self.expression.reduce()

    def run(self):
        while self.expression.reducible():
            print(self.expression)
            self.step()
        print(self.expression)
