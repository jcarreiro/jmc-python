# util.py
#
# Utility methods

import time

def elapsed_time(f, *args):
    t1 = time.clock()
    f(*args)
    t2 = time.clock()
    return t2 - t1
