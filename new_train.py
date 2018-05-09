# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import argparse
import time
import numpy as np
import tensorflow as tf

from zh_lib.dataset import ActionDataset
from zh_lib.load_data import load_info
from zh_lib.feed_queue import FeedQueue
import i3d 

_BATCH_SIZE = 6
_CLIP_SIZE = 66
_FRAME_SIZE = 224 
_LEARNING_RATE = 0.001
_GLOBAL_EPOCH = 40

_CHECKPOINT_PATHS = {
    'rgb': './data/checkpoints/rgb_scratch/model.ckpt',
    'flow': './data/checkpoints/flow_scratch/model.ckpt',
    #'rgb_imagenet': './data/checkpoints/rgb_imagenet/model.ckpt',
    'rgb_imagenet': './model/ucf101_rgb_0.941_model-14310',
    'flow_imagenet': './data/checkpoints/flow_imagenet/model.ckpt',
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

log_dir = './model'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def main(dataset_name, data_tag):
    assert data_tag in ['rgb', 'flow']
    # logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='w', format='%(message)s')


    train_info, test_info, class_num, _ = load_info(dataset_name, tag=data_tag)
    train_data = ActionDataset(dataset_name, class_num, train_info, 'frame{:06d}{:s}.jpg', tag=data_tag)
    test_data = ActionDataset(dataset_name, class_num, test_info, 'frame{:06d}{:s}.jpg', tag=data_tag)
    
    info1 = []
    for i in range(_GLOBAL_EPOCH):
        mid_tmp, _  =  train_data.gen_train_list(_BATCH_SIZE, _CLIP_SIZE, 1)
        info1.extend(mid_tmp)

    info2, _  =  test_data.gen_test_list()


    ##initialize
    #clip_holder    : none*none*frame_size*frame_size*channel
    #label_holder   : 1D
    #dropout_holder : scalar
    #is_train_holder: bool
    clip_holder = tf.placeholder(
        tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL[train_data.tag]])
    label_holder = tf.placeholder(tf.int32, [None])
    dropout_holder = tf.placeholder(tf.float32)
    is_train_holder = tf.placeholder(tf.bool)
    #tf.summary.image("demo", clip_holder[0], 6)
    #Forward pass
    with tf.variable_scope(_SCOPE[train_data.tag]):
        #insert i3d model
        model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
        #the line below outputs the final results with logits
        #__call__ uses _template, and _template uses _build when defined
        logits, _ = model(clip_holder, is_training=is_train_holder, dropout_keep_prob=dropout_holder)
        logits_dropout = tf.nn.dropout(logits, dropout_holder)
        #To change 400 classes to the ucf101 or hdmb classes   
        fc_out = tf.layers.dense(logits_dropout, _CLASS_NUM[dataset_name], tf.nn.relu, use_bias=True)
        #compute the top-k results for the whole batch size
        top_k_op = tf.nn.in_top_k(fc_out, label_holder, 1)

    variable_map = {}
    train_var = []
    # for variable in tf.global_variables():
    #     tmp = variable.name.split('/')
    #     #if tmp[1] == 'dense':
    #     #    train_var.append(variable)
    #     if tmp[0] == _SCOPE[train_data.tag] and tmp[1] != 'dense':
    #     #replace ':0' with '' and insert them into variable_map
    #         # if tmp[2] == 'Mixed_5c' and tmp[3] == 'Branch_1' and tmp[-2] == 'batch_norm' and tmp[4] != 'Conv3d_0a_1x1':
    #         #     continue
    #         variable_map[variable.name.replace(':0', '')] = variable
    #     #caculate l2-norm of these 'w' and 'kernel'
    #     if tmp[-1] == 'w:0' or tmp[-1] == 'kernel:0':
    #         weight_l2 = tf.nn.l2_loss(variable)
    #         tf.add_to_collection('weight_l2', weight_l2)
    
    #Start again from trainned model
    for variable in tf.global_variables():
        tmp = variable.name.split('/')
        print(variable.name)
        if tmp[0] == _SCOPE[train_data.tag]:
            variable_map[variable.name.replace(':0', '')] = variable
        #caculate l2-norm of these 'w' and 'kernel'
        if tmp[-1] == 'w:0' or tmp[-1] == 'kernel:0':
            weight_l2 = tf.nn.l2_loss(variable)
            tf.add_to_collection('weight_l2', weight_l2)        

    #import pre-trainned model
    saver = tf.train.Saver(var_list=variable_map, reshape=True)
    saver2 = tf.train.Saver(max_to_keep=10)
    #loss weight
    loss_weight = tf.add_n(tf.get_collection('weight_l2'), 'loss_weight')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_holder, logits=fc_out))
    #calculate total_loss
    total_loss = loss + 7e-7 * loss_weight
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_weight', loss_weight)
    tf.summary.scalar('total_loss', total_loss)
    #steps for training: the number of steps on batch per epoch
    per_epoch_step = int(np.ceil(train_data.size/_BATCH_SIZE))
    #global step constant
    global_step = _GLOBAL_EPOCH * per_epoch_step
    #global step counting
    global_index = tf.Variable(0, trainable=False)
    
    #Edited by Alex HU
    #decay_step = int(2*per_epoch_step)
    decay_step = 8000
    # learning_rate = tf.train.exponential_decay(
    #     _LEARNING_RATE, global_index, decay_step, 0.1, staircase=True)


    boundaries = [8000, 16000, 24000, 32000, 40000, 47000, 53000 ]
    values = [0.001, 0.0005, 0.0008, 3e-4, 6e-4, 1e-4, 3e-4, 7e-5]
    learning_rate = tf.train.piecewise_constant(global_index, boundaries, values)
    tf.summary.scalar('learning_rate', learning_rate)
    
    #FOR BATCH norm, we the use this updata_ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss, global_step=global_index)
    #initialize a seesion and TensorBoard
    sess = tf.Session()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, _CHECKPOINT_PATHS[train_data.tag+'_imagenet'])
    

    #Start Queue
    q = FeedQueue(queue_size=20)
    q.start_queue(train_data.get_batch, args=info1, process_num=7)



    #Train Process Begin !!!
    print(train_data.size)
    print('----Here we start!----')
    # logging.info('----Here we start!----')
    step = 0
    #for one epoch
    true_count = 0
    #for 20 batches
    tmp_count = 0
    accuracy_tmp = 0
    epoch_completed = 0

    while step <= global_step:
        step += 1
        start_time = time.time()
        clip, label = q.feed_me()
        #print(label)
        #preprocess the batch to have zero-mean and 1-variance   
        if train_data.tag == 'rgb':
            #clip = clip/255
            clip = 2*(clip/255)-1
        else:
            clip = 2*(clip/255)-1
        _, loss_now, loss_plus, top_1, summary = sess.run([optimizer, total_loss, loss_weight, top_k_op, merged],
                               feed_dict={clip_holder: clip,
                                          label_holder: label,
                                          dropout_holder: 0.36,
                                          is_train_holder: True})
        duration = time.time() - start_time
        tmp = np.sum(top_1)
        true_count += tmp
        tmp_count += tmp
        train_writer.add_summary(summary, step)
        # responsible for printing relevant results
        if step % 20 == 0:
            accuracy = tmp_count / (20*_BATCH_SIZE)
            print('step: %-4d, loss: %-.4f, accuracy: %.3f (%.2f sec/batch)' % (step, loss_now, accuracy, float(duration)))
            # logging.info('step: %-4d, loss: %-.4f, accuracy: %.3f (%.2f sec/batch)' % (step, loss_now, accuracy, float(duration)))
            tmp_count = 0
        
        if step % per_epoch_step ==0:
            epoch_completed = epoch_completed + 1
            accuracy = true_count/ (per_epoch_step*_BATCH_SIZE)
            print('Epoch%d, train accuracy: %.3f' % (epoch_completed, accuracy))
            # logging.info('Epoch%d, train accuracy: %.3f' % (train_data.epoch_completed, accuracy))
            true_count = 0
            if step % per_epoch_step == 0 and accuracy > 0.90:
                true_count = 0
                q2 = FeedQueue(queue_size=20)
                q2.start_queue(test_data.get_video, args=info2, process_num=6)                
                #start test process
                for i in range(test_data.size):
                    clip, label = q2.feed_me()

                    # clip, label = test_data.next_batch(
                    #     1, test_data.videos[i].total_frame_num, shuffle=False, data_augment=False)
                    if test_data.tag == 'rgb':
                        clip = 2*(clip/255)-1
                        #clip = clip/255
                    else:
                        clip = 2*(clip/255)-1
                    input_clip = clip[np.newaxis, :, :, :, :]
                    top_1 = sess.run(top_k_op,feed_dict={clip_holder: input_clip,
                                                            label_holder: label,
                                                            dropout_holder: 1,
                                                            is_train_holder: False})
                    true_count += np.sum(top_1)
                accuracy = true_count/ test_data.size
                true_count = 0
                #to ensure every test procedure has the same test size
                test_data.index_in_epoch = 0
                print('Epoch%d, test accuracy: %.3f' % (epoch_completed, accuracy))
                # logging.info('Epoch%d, test accuracy: %.3f' % (train_data.epoch_completed, accuracy))
                #saving the best params in test set
                if accuracy > 0.92:
                    if accuracy > accuracy_tmp:
                        accuracy_tmp = accuracy
                        saver2.save(sess,
                            os.path.join(log_dir, test_data.dataset_name+'_'+train_data.tag+'_{:.3f}_model'.format(accuracy)), step)
                q2.close_queue()
    train_writer.close()
    q.close_queue()
    sess.close()


if __name__ == '__main__':
    description = 'finetune on other dataset'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset_name', type=str)
    p.add_argument('data_tag', type=str)
    main(**vars(p.parse_args()))
