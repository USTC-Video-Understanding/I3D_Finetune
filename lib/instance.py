import os
import numpy as np
from PIL import Image
from lib.data_augment import transform_data, get_10_crop


class Video(object):
    def __init__(self, info_dict, img_format='{:06d}{:s}.jpg', tag='rgb'):
        '''
            info_dict_keys: name, path, length, label
            tag: 'rgb'(default) or 'flow'
            img_format: 'frame{:06d}{}.jpg'(default)
        '''
        self.name = info_dict['name']
        self.path = info_dict['path']
        self.total_frame_num = info_dict['length']
        self.label = info_dict['label']
        self.img_format = img_format
        self.tag = tag

    def load_img(self, index):
        img_dir = self.path
        if self.tag == 'rgb':
            return [Image.open(os.path.join(img_dir, self.img_format.format(index, ''))).convert('RGB')]
        elif self.tag == 'flow':
            u_img = Image.open(os.path.join(img_dir.format('u'), self.img_format.format(index, ''))).convert('L')
            v_img = Image.open(os.path.join(img_dir.format('v'), self.img_format.format(index, ''))).convert('L')
            return  [u_img, v_img]

    def __str__(self):
        return 'Video:\nname: {:s}\nframes: {:d}\nPath: {:s}\nLabel: {}'.format(
            self.name, self.total_frame_num, self.path, self.label)


class Video3D(Video):
    def get_frames(self, frame_num, start=1, sample=1, data_augment=False, random_start=False, side_length=224):
        '''
            return:
                frame_num * height * width * channel (rgb:3 , flow:2) 
        '''
        #assert frame_num <= self.total_frame_num
        start = start - 1
        if random_start:
            start = np.random.randint(max(self.total_frame_num-(frame_num-1)*sample, 1))
        frames = []
        for i in range(start, start+frame_num*sample, sample):
            frames.extend(self.load_img(i % self.total_frame_num + 1))
        frames = transform_data(frames, crop_size=side_length, random_crop=data_augment, random_flip=data_augment)
        frames_np = []
        if self.tag == 'rgb':
            for img in frames:
                frames_np.append(np.asarray(img))
        elif self.tag == 'flow':
            for i in range(0, len(frames), 2):
                tmp = np.stack([np.asarray(frames[i]), np.asarray(frames[i+1])], axis=2)
                frames_np.append(tmp)
        return np.stack(frames_np)
    
    def get_frames_10_crop(self, start, frame_num, side_length=224, sample=1):
        '''
            return:
                10 * frame_num * height * width * channel (rgb:3 , flow:2) 
        '''
        #assert frame_num <= self.total_frame_num
        frames = []
        for i in range(start, start+frame_num*sample, sample):
            frames.extend(self.load_img(i % self.total_frame_num + 1))
        ten_crop = get_10_crop(frames, crop_size=side_length)

        ten_crop_np = []
        if self.tag == 'rgb':
            for crop in ten_crop:
                crop_np = []
                for img in crop:
                    crop_np.append(np.asarray(img))
                ten_crop_np.append(crop_np)
        elif self.tag == 'flow':
            for crop in ten_crop:
                crop_np = []
                for i in range(0, len(crop), 2):
                    tmp = np.stack([np.asarray(crop[i]), np.asarray(crop[i+1])], axis=2)
                    crop_np.append(tmp)
                ten_crop_np.append(crop_np)
        return np.stack(ten_crop_np)

