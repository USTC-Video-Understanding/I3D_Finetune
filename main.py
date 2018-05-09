import os
import argparse
import time
import logging
from zh_lib.logging_tool import init_logging
from graph import train, val, test

_BATCH_SIZE = 10
_CLIP_SIZE = 32
_FRAME_SIZE = 224
_GLOBAL_EPOCH = 40
_LEARNING_RATE = 0.001
_DECAY_RATE = 0.7
_DECAY_EPOCH = 1
_WEIGHT_DECAY = 5e-7
_DROPOUT = 0.5

def parse_args():
    p = argparse.ArgumentParser(description='deep_ar')
    p.add_argument('dataset_name', type=str)
    p.add_argument('task', type=str)
    p.add_argument('-m', '--model', type=str, default='i3d')
    p.add_argument('-t', '--tag', type=str, default='rgb')
    p.add_argument('-o', '--out_type', type=str, default='sigmoid')
    p.add_argument('-bs', '--batch_size', type=int, default=_BATCH_SIZE)
    p.add_argument('-cs', '--clip_size', type=int, default=_CLIP_SIZE)
    p.add_argument('-s', '--sample', type=int, default=1)
    p.add_argument('-fs', '--frame_size', type=int, default=_FRAME_SIZE)
    p.add_argument('-lr', '--learning_rate', type=float, default=_LEARNING_RATE)
    p.add_argument('-dr', '--decay_rate', type=float, default=_DECAY_RATE)
    p.add_argument('-de', '--decay_epoch', type=int, default=_DECAY_EPOCH)
    p.add_argument('-d', '--dropout', type=float, default=_DROPOUT)
    p.add_argument('-wd', '--weight_decay', type=float, default=_WEIGHT_DECAY)
    p.add_argument('-ge', '--global_epoch', type=int, default=_GLOBAL_EPOCH)
    p.add_argument('-p', '--pretrained', action='store_true', default=False)
    p.add_argument('-cp', '--checkpoint_path', type=str, default='')
    parameter = p.parse_args()

    return  parameter

if __name__ == '__main__':
    p = parse_args()
    assert p.task in ['train', 'val', 'test']
    if p.task == 'train':
        log_dir = os.path.join('./train_model', time.strftime("%m_%d_%H_%M_%S", time.localtime()))
    elif p.task == 'val':
        log_dir = os.path.join('./val_result', time.strftime("%m_%d_%H_%M_%S", time.localtime()))
    elif p.task == 'test':
        log_dir = os.path.join('./test_result', time.strftime("%m_%d_%H_%M_%S", time.localtime()))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    init_logging(os.path.join(log_dir, 'log.txt'))

    items = list(vars(p).items())
    items.sort()
    for key, value in items:
        logging.info('{:}: {:}'.format(key, value))

    if p.task == 'train':
        train(p, log_dir)
    elif p.task == 'val':
        val(p, log_dir)
    elif p.task == 'test':
        test(p, log_dir)