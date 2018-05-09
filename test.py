# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import argparse
import time
import logging
import numpy as np
import tensorflow as tf

import i3d
from zh_lib.dataset import ActionDataset
from zh_lib.load_data import load_info
from zh_lib.feed_queue import FeedQueue
from zh_lib.label_trans import *

_FRAME_SIZE = 224 


_CHECKPOINT_PATHS = {
    'rgb': './model/ucf101_rgb_0.938_model-15900',
    'flow': './model/ucf101_flow_0.946_model-28620'
}



_CHANNEL = {
    'rgb': 3,
    'flow': 2,
}

_SCOPE = {
    'rgb': 'RGB',
    'flow': 'Flow',
}

_CLASS_NUM = {
    'ucf101': 101,
    'hmdb51': 51
}

log_dir = 'error_record'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

def main(dataset_name, data_tag):
    assert data_tag in ['rgb', 'flow', 'mixed']

    # logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, 'log_'+data_tag+'.txt'), filemode='w', format='%(message)s')

    label_map = get_label_map(os.path.join('data', dataset_name, 'label_map.txt'))


    _, test_info, class_num, _ = load_info(dataset_name, tag='rgb')
    _, test_info1, _, _ = load_info(dataset_name, tag='flow')

    label_holder = tf.placeholder(tf.int32, [None])
    if data_tag in ['rgb', 'mixed']:
        rgb_data = ActionDataset(dataset_name, class_num, test_info, 'frame{:06d}{:s}.jpg', tag='rgb')      
        rgb_holder = tf.placeholder(
            tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['rgb']])
    if data_tag in ['flow', 'mixed']:
        flow_data = ActionDataset(dataset_name, class_num, test_info1, 'frame{:06d}{:s}.jpg', tag='flow')
        flow_holder = tf.placeholder(
            tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['flow']])
    info1, _  =  rgb_data.gen_test_list()
    info2, _  =  flow_data.gen_test_list()



    #insert the model
    if data_tag in ['rgb', 'mixed']:
        with tf.variable_scope(_SCOPE['rgb']):
            rgb_model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(rgb_holder, is_training=False, dropout_keep_prob=1)
            rgb_logits_dropout = tf.nn.dropout(rgb_logits, 1)   
            rgb_fc_out = tf.layers.dense(rgb_logits_dropout, _CLASS_NUM[dataset_name], tf.nn.relu, use_bias=True)
            rgb_top_k_op = tf.nn.in_top_k(rgb_fc_out, label_holder, 1)
    if data_tag in ['flow', 'mixed']:
        with tf.variable_scope(_SCOPE['flow']):
            flow_model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
            flow_logits, _ = flow_model(flow_holder, is_training=False, dropout_keep_prob=1)
            flow_logits_dropout = tf.nn.dropout(flow_logits, 1)   
            flow_fc_out = tf.layers.dense(flow_logits_dropout, _CLASS_NUM[dataset_name], tf.nn.relu, use_bias=True)
            flow_top_k_op = tf.nn.in_top_k(flow_fc_out, label_holder, 1)

    #construct two separate feature map and saver(rgb_saver,flow_saver)
    variable_map = {}
    if data_tag in ['rgb', 'mixed']:
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == _SCOPE['rgb']:
                variable_map[variable.name.replace(':0', '')]=variable
        rgb_saver = tf.train.Saver(var_list=variable_map)
    variable_map = {}
    if data_tag in ['flow', 'mixed']:
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == _SCOPE['flow']:
                variable_map[variable.name.replace(':0', '')]=variable
        flow_saver = tf.train.Saver(var_list=variable_map, reshape=True)


    ##Edited Version by AlexHu
    if data_tag == 'rgb':
        fc_out = rgb_fc_out   
        softmax = tf.nn.softmax(fc_out)
    if data_tag == 'flow':
        fc_out = flow_fc_out   
        softmax = tf.nn.softmax(fc_out)
    if data_tag == 'mixed':
        fc_out = 0.5*rgb_fc_out + 0.5*flow_fc_out
        softmax = tf.nn.softmax(fc_out)
    top_k_op = tf.nn.in_top_k(softmax, label_holder, 1)

    #GPU config
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    # sess = tf.Session(config=config)# config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    # sess = tf.Session(config=config)
    
    #start a new session and restore the fine-tuned model
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if data_tag in ['rgb', 'mixed']:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
    if data_tag in ['flow', 'mixed']:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])

    #Start Queue
    rgb_queue = FeedQueue(queue_size=20)
    rgb_queue.start_queue(rgb_data.get_video, args=info1, process_num=5)
    flow_queue = FeedQueue(queue_size=20)
    flow_queue.start_queue(flow_data.get_video, args=info2, process_num=5)


    #Here we start the test procedure
    print('----Here we start!----')
    true_count = 0
    video_size = len(test_info)
    error_record = open(os.path.join(log_dir, 'error_record_'+data_tag+'.txt'), 'w')
    rgb_fc_data = np.zeros((video_size, _CLASS_NUM[dataset_name]))
    flow_fc_data = np.zeros((video_size, _CLASS_NUM[dataset_name]))
    label_data = np.zeros((video_size,1))
    
    #just load 1 video for test,this place needs to be improved
    for i in range(video_size):
        print(i)
        if data_tag in ['rgb', 'mixed']:
            rgb_clip, label = rgb_queue.feed_me()
            rgb_clip = rgb_clip/255
            input_rgb = rgb_clip[np.newaxis, :, :, :, :]
            video_name = rgb_data.videos[i].name
        if data_tag in ['flow', 'mixed']:
            flow_clip, label = flow_queue.feed_me()
            flow_clip = 2*(flow_clip/255)-1
            input_flow = flow_clip[np.newaxis, :, :, :, :]
            video_name = flow_data.videos[i].name
        input_label = np.array([label])

        #Extract features from rgb and flow
        top_1, predictions, curr_rgb_fc_data, curr_flow_fc_data= sess.run([top_k_op, fc_out, rgb_fc_out, flow_fc_out],
            feed_dict={ rgb_holder: input_rgb,
                        flow_holder: input_flow,
                        label_holder: input_label})
        rgb_fc_data[i, :] = curr_rgb_fc_data
        flow_fc_data[i,:] = curr_flow_fc_data
        label_data[i,:] =  label       
       
        tmp = np.sum(top_1)
        true_count += tmp
        print('Video%d: %d, accuracy: %.4f (%d/%d) , name:%s' % (i+1, tmp, true_count/video_size, true_count, video_size, video_name))
        # logging.info('Video%d: %d, accuracy: %.4f (%d/%d) , name:%s' % (i+1, tmp, true_count/video_size, true_count, video_size, video_name))

        #self_added
        print(predictions[0,np.argmax(predictions, axis=1)[0]])
        print(trans_label(np.argmax(predictions, axis=1)[0], label_map))
        #print(np.argmax(label))
        #print(trans_label(np.argmax(label), label_map))

        if tmp==0:
            wrong_answer = np.argmax(predictions, axis=1)[0]
            #Attention: the graph output are converted into the type of numpy.array
            print('---->answer: %s, probability: %.2f' % (trans_label(wrong_answer, label_map), predictions[0, wrong_answer]))
            # logging.info('---->answer: %s, probability: %.2f' % (trans_label(wrong_answer, label_map), predictions[0, wrong_answer]))
            error_record.write(
                'video: %s, answer: %s, probability: %.2f\n' % (video_name, trans_label(wrong_answer, label_map), predictions[0, wrong_answer]))
    error_record.close()
    accuracy = true_count/ video_size
    print('test accuracy: %.4f' % (accuracy))
    # logging.info('test accuracy: %.4f' % (accuracy))
    np.save('obj_{}_rgb_fc_{}.npy'.format(dataset_name, accuracy), rgb_fc_data)
    np.save('obj_{}_flow_fc_{}.npy'.format(dataset_name, accuracy), flow_fc_data)
    np.save('obj_{}_label.npy'.format(dataset_name),label_data)


    rgb_queue.close_queue()
    flow_queue.close_queue()
    sess.close()


if __name__ == '__main__':
    description = 'finetune on other dataset'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset_name', type=str)
    p.add_argument('data_tag', type=str)
    main(**vars(p.parse_args()))
