import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from zh_lib.dataset import ActionDataset
from zh_lib.load_data import load_info
from zh_lib.feed_queue import FeedQueue
from zh_lib.model.i3d import I3D


def train(p, log_dir):
    train_info, test_info, class_num, img_format = load_info(p.dataset_name, tag=p.tag)
    train_data = ActionDataset(p.dataset_name, class_num, train_info, img_format, tag=p.tag)
    test_data = ActionDataset(p.dataset_name, class_num, test_info, img_format, tag=p.tag)
    if p.tag == 'rgb':
        input_channel = 3 
    elif p.tag == 'flow':
        input_channel = 2
    model = I3D(num_classes=class_num, dropout_keep_prob=p.dropout, input_channel=input_channel)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=p.learning_rate,
                          momentum=0.9,
                          weight_decay=p.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(p.global_epoch):
        info, _ = train_data.gen_train_list(p.batch_size, p.clip_size, p.sample)
        train_queue = FeedQueue(queue_size=20)
        train_queue.start_queue(train_data.get_batch, args=info, process_num=5)



        train_queue.close_queue()
        time.sleep(5)


def val():
    pass


def test():
    pass


