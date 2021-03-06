import math

# Converts from WGS84 coordinates into pixels, assuming a spherical Mercator
# projection with a circumference of 512 px at level 1, doubling in size for
# each subsequent zoom level.
def wgs84ToPixels(lat, lon, z):
    r = 2 ** (z + 7) / math.pi
    x = r * math.radians(lon)
    y = r * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return (x, y)
