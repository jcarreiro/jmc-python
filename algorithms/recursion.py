# Any recursive function can be transformed into an iterative function which
# uses an explicit stack. Here's an example:

# Basic stack class.
class Stack(object):
    def __init__(self):
        self.s = []

    def __len__(self):
        return len(self.s)

    def __repr__(self):
        return '[' + ', '.join([str(x) for x in self.s]) + ']'

    def top(self):
        return self.s[-1]

    def push(self, v):
        self.s.append(v)

    def pop(self):
        return self.s.pop()

# This is the iterative version of the function, which uses an explicit stack.
def filter_args_iter(defs):
    # Helper class for holding the state at each "frame".
    class Node(object):
        def __init__(self, defs, x, values):
            self.defs = defs
            self.x = x
            self.values = values

        def __repr__(self):
            return 'Node({0}, {1}, {2})'.format(self.defs, self.x, self.values)

    r = []
    s = Stack()
    s.push(Node(defs, None, {}))
    while s:
        t = s.pop()
        if t.defs:
            # "Recurse" on the next possible value of the first remaining param.
            p = t.defs[0]
            v = None
            if t.x == None:
                v = p.start
            else:
                v = t.x + p.step
            if v < p.stop:
                s.push(Node(t.defs, v, t.values))
                s.push(Node(t.defs[1:], None, merge_dicts(t.values, {p.name: v})))
        else:
            # Otherwise, we're at a leaf node so yield the values and "return".
            r.append(t.values)
    return r

# This is the recursive version of the function.
def filter_args(defs):
    def helper(defs, values, r):
        if not defs:
            # All done, return this set of values
            r.append(values)
        else:
            # Fix value of next parameter and recurse on remaining.
            p = defs[0]
            for x in np.arange(p.start, p.stop, p.step):
                helper(defs[1:], merge_dicts(values, {p.name : x}), r)
    r = []
    helper(defs, {}, r)
    return r
