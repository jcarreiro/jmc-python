import urlparse

def urldecode(qs):
    return urlparse.parse_qs(qs)
