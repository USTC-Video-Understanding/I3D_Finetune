from __future__ import division
import random
import numpy as np
from PIL import ImageOps


def transform_data(data, crop_size=224, random_crop=True, random_flip=False):
    #length is the pic' width, when width is pic's height
    length = data[0].size[0]
    width = data[0].size[1]
    if random_crop:
        #randomly select the crop area
        x0 = random.randint(0, length - crop_size)
        y0 = random.randint(0, width - crop_size)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        for i, img in enumerate(data):
            data[i] = img.crop((x0, y0, x1, y1))
    else:
        x0 = int((length-crop_size)/2)
        y0 = int((width-crop_size)/2)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        for i, img in enumerate(data):
            data[i] = img.crop((x0, y0, x1, y1))
    if random_flip and x0%2 == 0:
        for i, img in enumerate(data):
            data[i] = ImageOps.mirror(img)
    return  data
