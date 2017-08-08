# /usr/local/bin/python3
#
# Plot data over an OSM map.

import io
import pandas
import matplotlib.pyplot as plt
import numpy as np
import math
import urllib

from mpl_toolkits.basemap import Basemap
from PIL import Image

def deg2num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
  return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
  """
  http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
  This returns the NW-corner of the square.
  Use the function with xtile+1 and/or ytile+1 to get the other corners.
  With xtile+0.5 & ytile+0.5 it will return the center of the tile.
  """
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return (lat_deg, lon_deg)

def getImageCluster(lat_deg, lon_deg, delta_lat,  delta_long, zoom):
    smurl = r"http://a.tile.openstreetmap.org/{0}/{1}/{2}.png"
    xmin, ymax = deg2num(lat_deg, lon_deg, zoom)
    xmax, ymin = deg2num(lat_deg + delta_lat, lon_deg + delta_long, zoom)

    bbox_ul = num2deg(xmin, ymin, zoom)
    bbox_ll = num2deg(xmin, ymax + 1, zoom)
    #print bbox_ul, bbox_ll

    bbox_ur = num2deg(xmax + 1, ymin, zoom)
    bbox_lr = num2deg(xmax + 1, ymax +1, zoom)
    #print bbox_ur, bbox_lr

    cluster = Image.new('RGB',((xmax-xmin+1)*256-1,(ymax-ymin+1)*256-1) )
    for xtile in range(xmin, xmax+1):
        for ytile in range(ymin,  ymax+1):
            try:
                imgurl=smurl.format(zoom, xtile, ytile)
                print("Opening: " + imgurl)
                imgstr = urllib.request.urlopen(imgurl).read()
                tile = Image.open(io.StringIO(imgstr))
                cluster.paste(tile, box=((xtile-xmin)*255 ,  (ytile-ymin)*255))
            except:
                print("Couldn't download image")
                tile = None

    return cluster, [bbox_ll[1], bbox_ll[0], bbox_ur[1], bbox_ur[0]]

def plot_map():
    df = pandas.read_csv('/Users/jcarreiro/Downloads/badid.csv', sep='\t', header=-1)
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    bbox = df[3].min(), df[4].min(), df[3].max(), df[4].max()
    m = Basemap(llcrnrlat=bbox[0],
                llcrnrlon=bbox[1],
                urcrnrlat=bbox[2],
                urcrnrlon=bbox[3],
                rsphere=(6378137.00, 6356752.3142),
                resolution='h',
                projection='merc',
                lat_ts=20.0)
    m.drawcoastlines()
    m.fillcontinents()
    m.drawparallels(np.arange(bbox[0], bbox[2], 0.1), labels=[1, 0, 0, 1])
    m.drawmeridians(np.arange(bbox[1], bbox[3], 0.1), labels=[1, 0, 0, 1])
    m.plot(df[4].values, df[3].values, latlon=True)
    plt.show()
