from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import string
import random

# Simple recursive descent parser for dice rolls, e.g. '3d6+1d8+4'.
#
#   roll := die {('+' | '-') die} ('+' | '-') modifier
#   die := number 'd' number
#   modifier := number

class StringBuf(object):
    def __init__(self, s):
        self.s = s
        self.pos = 0

    def peek(self):
        return self.s[self.pos]

    def getc(self):
        c = self.peek()
        self.pos += 1
        return c

    def ungetc(self):
        self.pos -= 1

    def tell(self):
        return self.pos

class Symbol(object):
    NUMBER = 0
    D = 1
    PLUS = 2
    MINUS = 3

    def __init__(self, type_, pos, value)

def next_symbol(s):
    c = s.getc()
    while c in string.whitespace:
        c = s.getc()
    if c in string.digits:
        # start of a number
        literal = c
        c = s.getc()
        while c in string.digits:
            literal += c
            c = s.getc()
        s.ungetc()
        sym = (Symbol.NUMBER,
    elif c == 'd':
        # die indicator
        pass
    elif c == '+':
        # plus sign
        pass
    elif c == '-':
        # minus sign
        pass
    else:
        # unrecognized input
        raise ValueError('Syntax error at position ' + s.tell())
    return ()
