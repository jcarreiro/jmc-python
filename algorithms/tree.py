def visible_r(t, max_):
    if t == None:
        return 0

    if t[0] >= max_:
        return 1 + visible_r(t[1], t[0]) + visible_r(t[2], t[0])
    
    return visible_r(t[1], max_) + visible_r(t[2], max_)

def visible(t):
    """Returns the number of 'visible' nodes in a tree. A node is considered
    to be 'visible' if none of its ancestors have a greater value."""
    return visible_r(t, t[0])
