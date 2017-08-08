from __future__ import absolute_import

import random

from jmc.algorithms.searching import binary_search

def coin_flip(seed=None):
  """Generates coin flips using a fair six-sided die."""
  if seed:
    random.seed(seed)
    return random.randint(1, 6) > 3

def test_coin_flip(count):
  heads = 0
  tails = 0
  for x in xrange(0, count):
    if coin_flip():
      heads += 1
    else:
      tails += 1
  return [heads, tails]

class BiasedDie(object):
  def __init__(self, faces, r=random.Random()):
    """Create a biased die. Faces must be a list of floats, which are the
    cumulative probability of a roll resulting in a value less than or equal to
    the value of that face. Faces are implictly numbered from 1 to N.
    """
    self.faces = faces
    self.r = r

  def roll(self):
    return binary_search(self.faces, r.random()) + 1

def fair_d6(seed=None):
    """Uses a biased d6 to generate fair values between 1 and 6."""
    # pick random weights for the faces, then normalize
    if seed:
      random.seed(seed)
    faces = [random.random() for x in range(6)]
    total = sum(faces)
    faces = map(lambda x: x / total, faces)
    faces = [sum(faces[:x]) for x in range(1,7)]
    print faces

    # Roll a biased d6. If we see a 1, 2, or 3, followed by a 4, 5, or 6, call
    # that a 0, and call a 4, 5, or 6 followed by a 1, 2, or 3 a 1. Ignore all
    # other results. This gives us a 0 or a 1 with equal probability.
    d6 = BiasedDie(faces, r) # ok to re-use r, we're done with the stream now
    while True:
      s = '0b' # yeah this is clowny
      while len(s) < 5:
        a = d6.roll()
        b = d6.roll()
        if a <= 3 and b >= 4:
          s += '0'
        elif a >= 4 and b <= 3:
          s += '1'
      result = int(s, 0)
      if result > 0 and result < 7:
        yield result
