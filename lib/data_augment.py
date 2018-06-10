from __future__ import division
import random
from PIL import ImageOps


def transform_data(data, scale_size=256, crop_size=224, random_crop=False, random_flip=False):
    data = resize(data, scale_size)
    width = data[0].size[0]
    height = data[0].size[1]
    if random_crop:
        x0 = random.randint(0, width - crop_size)
        y0 = random.randint(0, height - crop_size)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        for i, img in enumerate(data):
            data[i] = img.crop((x0, y0, x1, y1))
    else:
        x0 = int((width-crop_size)/2)
        y0 = int((height-crop_size)/2)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        for i, img in enumerate(data):
            data[i] = img.crop((x0, y0, x1, y1))
    if random_flip and random.randint(0,1) == 0:
        for i, img in enumerate(data):
            data[i] = ImageOps.mirror(img)
    return  data

def get_10_crop(data, scale_size=256, crop_size=224):
    data = resize(data, scale_size)
    width = data[0].size[0]
    height = data[0].size[1]
    top_left = [[0, 0],
                [width-crop_size, 0],
                [int((width-crop_size)/2), int((height-crop_size)/2)],
                [0, height-crop_size],
                [width-crop_size, height-crop_size]]
    crop_data = []
    for point in top_left:
        non_flip = []
        flip = []
        x_0 = point[0]
        y_0 = point[1]
        x_1 = x_0 + crop_size
        y_1 = y_0 + crop_size
        for img in data:
            tmp = img.crop((x_0, y_0, x_1, y_1))
            non_flip.append(tmp)
            flip.append(ImageOps.mirror(tmp))
        crop_data.append(non_flip)
        crop_data.append(flip)
    return  crop_data

def scale(data, scale_size):
    width = data[0].size[0]
    height = data[0].size[1]
    if (width==scale_size and height>=width) or (height==scale_size and width>=height):
        return data
    if width >= height:
        h = scale_size
        w = round((width/height)*scale_size)
    else:
        w = scale_size
        h = round((height/width)*scale_size)
    for i, image in enumerate(data):
        data[i] = image.resize((w, h))
    return  data

def resize(data, scale_size):
    width = data[0].size[0]
    height = data[0].size[1]
    if (width==scale_size and height>=width) or (height==scale_size and width>=height):
        return data
    for i, image in enumerate(data):
        data[i] = image.resize((scale_size, scale_size))
    return  data
