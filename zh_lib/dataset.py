from __future__ import division
import math
import numpy as np
from zh_lib.instance import Video3D


class Dataset(object):
    def __init__(self, name, total_class_num):
        self.dataset_name = name
        self.total_class_num = total_class_num


class ActionDataset(Dataset):
    def __init__(self, name, total_class_num, video_info, img_format, tag='rgb'):
        super().__init__(name, total_class_num)
        self.tag = tag
        self.videos = [Video3D(x, img_format, tag) for x in video_info]
        self.size = len(self.videos)

    def gen_train_list(self, batch_size, frame_num, sample=1):
        '''
            def get_frames(self, frame_num, start=1, sample=1, 
                data_augment=False, random_start=False, side_length=224):
        '''
        train_list = []
        video_num_per_epoch = math.ceil(self.size/batch_size) * batch_size
        perm = np.arange(video_num_per_epoch) % self.size
        np.random.shuffle(perm)
        for j in range(0, video_num_per_epoch, batch_size):
            batch = []
            for k in range(batch_size):
                idx = perm[j+k]
                info = [perm[idx], frame_num, 1, sample, True, True]
                batch.append(info)
            train_list.append((batch,))
        return train_list, video_num_per_epoch

    def get_batch(self, clip_infos):
        batch = []
        label = []
        for info in clip_infos:
            batch.append(self.videos[info[0]].get_frames(*info[1:]))
            label.append(self.videos[info[0]].label)
        return np.stack(batch), np.stack(label)

    def gen_test_list(self, sample=1):
        test_list = []
        name = []
        for i, video in enumerate(self.videos):
            test_list.append(([i, 251, 1, sample, False, False],))
            name.append(video.name)
        return test_list, name

    def get_video(self, info):
        video = self.videos[info[0]].get_frames(*info[1:])
        label = self.videos[info[0]].label
        return video, np.array([label])

    # def gen_train_list(self, batch_size, total_epoch, frame_num, sample=1):
    #     '''
    #         def get_frames(self, frame_num, start=1, sample=1, 
    #             data_augment=False, random_start=False, side_length=224):
    #     '''
    #     train_list = []
    #     video_num_per_epoch = math.ceil(self.size/batch_size) * batch_size
    #     perm = np.arange(video_num_per_epoch) % self.size
    #     for i in range(total_epoch):
    #         np.random.shuffle(perm)
    #         for j in range(0, video_num_per_epoch, batch_size):
    #             batch = []
    #             for k in range(batch_size):
    #                 idx = perm[j+k]
    #                 info = [perm[idx], frame_num, 1, sample, True, True]
    #                 batch.append(info)
    #             train_list.append((batch,))
    #     return train_list, video_num_per_epoch