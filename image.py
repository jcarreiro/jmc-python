#!/usr/local/bin/python

import datetime
import operator
import PIL.Image
import PIL.ExifTags

def parse_timestamp(ts):
    return datetime.datetime.strptime(ts, '%Y:%m:%d %H:%M:%S')

def get_exif(img_path):
    try:
      img = PIL.Image.open(img_path)
      return {
        PIL.ExifTags.TAGS[k] : v for k, v in img._getexif().items() if k in PIL.ExifTags.TAGS
      }
    except:
      return {}

def order_by_exif_date(img_paths, reverse=False):
    exifs = { path: get_exif(path) for path in img_paths }
    return zip(
        *sorted(
            exifs.items(),
            key=lambda (k, v): parse_timestamp(v['DateTime']),
            reverse=reverse,
        )
    )[0]
