import os
import pandas as pd


_CLASS_NUM = {
    'ucf101': 101,
    'hmdb51': 51,
    'something': 0,
    'kinetics': 400,
}

_IMG_FORMAT = {
    'ucf101': 'frame{:06d}{:s}.jpg',
}

def load_info(name, task='train', tag='rgb'):
    if name == 'ucf101':
        train_info, val_info = UCF101.get_info(tag)
        return train_info, val_info, _CLASS_NUM[name], _IMG_FORMAT[name]
    elif name == 'hmdb51':
        pass
    elif name == 'something':
        pass
    elif name == 'kinetics':
        pass


class UCF101():
    @staticmethod
    def get_info(tag='rgb'):
        base_dir = {
            'rgb': '/data4/zhouhao/dataset/ucf101/jpegs_256',
            'flow': '/data4/zhouhao/dataset/ucf101/tvl1_flow/{:s}',
        }
        data_dir = 'data/ucf101/'+tag+'.txt'
        test_split = 'data/ucf101/testlist01.txt'
        f_1 = open(data_dir, 'r')
        test = [x.split('/')[1].split('.')[0] for x in open(test_split, 'r').readlines()]
        train_info = []
        test_info = []
        for line in f_1.readlines():
            line = line.strip().split(' ')
            info = {'name': line[0],
                    'path': os.path.join(base_dir[tag], line[1]),
                    'length': int(line[2]),
                    'label': int(line[3])}
            if line[0] in test:
                test_info.append(info)
            else:
                train_info.append(info)
        f_1.close()
        return train_info, test_info


def split_data(data_info, test_split):
    f_1 = open(data_info, 'r')
    f_2 = open(test_split, 'r')
    test = []
    train_info = []
    test_info = []
    for line in f_2.readlines():
        test.append(line.split('/')[1].split('.')[0])
    for line in f_1.readlines(): 
        line = line.strip().split(' ')
        info = {'name': line[0],
                'path': line[1],
                'length': int(line[2]),
                'label': int(line[3])}
        if line[0] in test:
            test_info.append(info)
        else:
            train_info.append(info)
    f_1.close()
    f_2.close()
    return train_info, test_info

def load_info_from_csv(csv_dir, data_dir):
    file = pd.read_csv(csv_dir).to_dict('index')
    infos = []
    for idx in file.keys():
        file[idx]['path'] = os.path.join(data_dir, file[idx]['name'])
        infos.append(file[idx])
    return infos
