# Memoizes a function.
#
# From wiki.python.org/moin/PythonDecoratorLibrary.
class memoized(object):
    def __init__(self, f):
        self.f = f
        self.cache = {}

    def __call__(self, *args):
        try:
            return self.cache[args]
        except KeyError:
            value = self.f(*args)
            self.cache[args] = value
            return value
        except TypeError:
            # This can occur if the arguments include a type that can't be used
            # as a key in a dictionary, such as a list.
            return self.f(*args)

# Counts the number of calls to a function.
#
# From wiki.python.org/moin/PythonDecoratorLibrary.
class countcalls(object):
    instances = {}
    def __init__(self, f):
        self.f = f
        self.count = 0
        countcalls.instances[f] = self

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.f(*args, **kwargs)

    @staticmethod
    def count(f):
        return countcalls.instances[f].count

    @staticmethod
    def counts():
        return dict([(f, countcalls.count(f)) for f in countcalls.instances])
