from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import numpy as np
import tensorflow as tf

import i3d 
from rmb_lib.action_dataset import *


_BATCH_SIZE = 6
_CLIP_SIZE = 64
_FRAME_SIZE = 224 
_LEARNING_RATE = 0.001
_GLOBAL_EPOCH = 40
_prefetch_buffer_size = 30

_CHECKPOINT_PATHS = {
    'rgb': './data/checkpoints/rgb_scratch/model.ckpt',
    'flow': './data/checkpoints/flow_scratch/model.ckpt',
    #'rgb_imagenet': './data/checkpoints/rgb_imagenet/model.ckpt',
    'rgb_imagenet': './data/checkpoints/rgb_imagenet/ucf101_rgb_0.945_model-49290',
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


def _parse_function(train_info_tensor):
    #print(train_info_tensor) 
    clip_holder, label_holder = tf.py_func(single_video, [train_info_tensor], [tf.float32, tf.int64])
    return clip_holder, label_holder

def _parse_function2(test_info_tensor):
    #print(train_info_tensor) 
    clip_holder, label_holder = tf.py_func(single_video2, [test_info_tensor], [tf.float32, tf.int64])
    return clip_holder, label_holder


def single_video(train_info):
    train_data = Action_Dataset('ucf101', 'rgb', [train_info])
    clip_seq, label_seq = train_data.next_batch(1, _CLIP_SIZE)
    clip_seq = 2*(clip_seq/255) - 1 
    clip_seq = np.array(clip_seq, dtype = 'float32')  
    return clip_seq, label_seq

def single_video2(test_info):
    test_data = Action_Dataset('ucf101', 'rgb', [test_info])
    clip_seq, label_seq = test_data.next_batch(1, 251, shuffle=False, data_augment=False)
    clip_seq = 2*(clip_seq/255) - 1 
    clip_seq = np.array(clip_seq, dtype = 'float32')  
    return clip_seq, label_seq


def main(dataset_name, data_tag):
    assert data_tag in ['rgb', 'flow']
    # logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='w', format='%(message)s')


#    with tf.device('/cpu:0'):
    ##  Data Preload  ###
    train_info, test_info = split_data(
        os.path.join('./data', dataset_name, data_tag+'.txt'),
        os.path.join('./data', dataset_name, 'testlist01.txt'))
    train_data = Action_Dataset(dataset_name, data_tag, train_info)
    test_data = Action_Dataset(dataset_name, data_tag, test_info)
 

    ## Every element in train_info is shown as below:
    ## ['v_ApplyEyeMakeup_g08_c01', '/data4/zhouhao/dataset/ucf101/jpegs_256/v_ApplyEyeMakeup_g08_c01', '121', '0']
    ## len train_info:9537; len test_info:3783
    train_info_tensor = tf.constant(train_info)
    test_info_tensor  = tf.constant(test_info)

    ###  Dataset building
    ##   Phase 1 Trainning
    # one element in this dataset is (train_info list)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_info_tensor))
    # one element in this dataset is (single image_postprocess, single label)
    
    # one element in this dataset is (batch image_postprocess, batch label)
    train_dataset = train_dataset.shuffle(buffer_size=9540)
    train_dataset = train_dataset.map(_parse_function, num_parallel_calls=7)
    train_dataset = train_dataset.repeat().batch(_BATCH_SIZE) 
    train_dataset = train_dataset.prefetch(buffer_size=_prefetch_buffer_size)


    ##   Phase 2 Testing
    # one element in this dataset is (train_info list)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_info_tensor))
    # one element in this dataset is (single image_postprocess, single label)
    test_dataset = test_dataset.map(_parse_function2, num_parallel_calls=10) 
    # one element in this dataset is (batch image_postprocess, batch label)
    test_dataset = test_dataset.batch(1).repeat()
    test_dataset = test_dataset.prefetch(buffer_size=_prefetch_buffer_size)    

    # iterator = dataset.make_one_shot_iterator()
    # clip_holder, label_holder = iterator.get_next()
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op  = iterator.make_initializer(test_dataset)

    clip_holder, label_holder = iterator.get_next()
    clip_holder  = tf.squeeze(clip_holder,  [1])
    label_holder = tf.squeeze(label_holder, [1])
    if data_tag == 'rgb': 
        clip_holder.set_shape([None, None, 224, 224, 3])
    else:
        clip_holder.set_shape([None, None, 224, 224, 2])
    dropout_holder  = tf.placeholder(tf.float32)
    is_train_holder = tf.placeholder(tf.bool)


    # with tf.Session() as sess:
    #     #sess.run(init_op)
    #     try:
    #         sess.run(train_init_op)
    #         for i in range(10):
    #             clip, label = sess.run([clip_holder,label_holder])
    #             print(clip.shape, label)
    #             print('aa')
    #         sess.run(test_init_op)
    #         for i in range(10):   
    #             clip, label = sess.run([clip_holder,label_holder])
    #             print(clip.shape, label)
    #             print('bb')
    #     except tf.errors.OutOfRangeError:
    #         print("end!")    
    # assert 1==2


    #inference module
    ###  Inference Module
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


    ###    Loss calculation, including L2-norm   
    variable_map = {}
    train_var = []
    for variable in tf.global_variables():
        tmp = variable.name.split('/')
        if tmp[0] == _SCOPE[train_data.tag]:
            variable_map[variable.name.replace(':0', '')] = variable
        if tmp[-1] == 'w:0' or tmp[-1] == 'kernel:0':
            weight_l2 = tf.nn.l2_loss(variable)
            tf.add_to_collection('weight_l2', weight_l2)
    loss_weight = tf.add_n(tf.get_collection('weight_l2'), 'loss_weight')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_holder, logits=fc_out))
    total_loss = loss + 9e-7 * loss_weight
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_weight', loss_weight)
    tf.summary.scalar('total_loss', total_loss)


    ###   Import Pre-trainned model 
    saver = tf.train.Saver(var_list=variable_map, reshape=True)
    saver2 = tf.train.Saver(max_to_keep=10)
    ###  Specific Hyperparams
    #steps for training: the number of steps on batch per epoch
    per_epoch_step = int(np.ceil(train_data.size/_BATCH_SIZE))
    #global step constant
    global_step = _GLOBAL_EPOCH * per_epoch_step
    #global step counting
    global_index = tf.Variable(0, trainable=False)

    # decay_step = 8000
    # learning_rate = tf.train.exponential_decay(
    #     _LEARNING_RATE, global_index, decay_step, 0.1, staircase=True)
    # tf.summary.scalar('learning_rate', learning_rate)


    boundaries = [8000,  16000,  24000, 31000, 40000, 50000, 60000, 75000] 
    values     = [0.0008, 0.0004, 0.0005, 7e-5,  5e-5,  5e-6,  2e-6,  1e-7 ]
    learning_rate = tf.train.piecewise_constant(global_index, boundaries, values)
    tf.summary.scalar('learning_rate', learning_rate)


    ###    Optimizer set-up
    #FOR BATCH norm, we the use this updata_ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss, global_step=global_index)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(train_init_op) 
    saver.restore(sess, _CHECKPOINT_PATHS[train_data.tag+'_imagenet'])
    
    
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
  
        _, loss_now, loss_plus, top_1, summary = sess.run([optimizer, total_loss, loss_weight, top_k_op, merged],
                               feed_dict={dropout_holder: 0.25,
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
            epoch_completed += 1
            accuracy = true_count/ (per_epoch_step*_BATCH_SIZE)
            print('Epoch%d, train accuracy: %.3f' % (epoch_completed, accuracy))
            # logging.info('Epoch%d, train accuracy: %.3f' % (train_data.epoch_completed, accuracy))
            true_count = 0
            if step % per_epoch_step == 0 and accuracy > 0.85:
                sess.run(test_init_op) 
                true_count = 0
                #start test process
                print(test_data.size)
                for i in range(test_data.size):
                    #print(i,true_count)
                    top_1 = sess.run(top_k_op,feed_dict={dropout_holder: 1,
                                                         is_train_holder: False})
                    true_count += np.sum(top_1)
                accuracy = true_count/ test_data.size
                true_count = 0
                #to ensure every test procedure has the same test size
                test_data.index_in_epoch = 0
                print('Epoch%d, test accuracy: %.3f' % (epoch_completed, accuracy))
                # logging.info('Epoch%d, test accuracy: %.3f' % (train_data.epoch_completed, accuracy))
                #saving the best params in test set
                if accuracy > 0.85:
                    if accuracy > accuracy_tmp:
                        accuracy_tmp = accuracy
                        saver2.save(sess,
                            os.path.join(log_dir, test_data.dataset_name+'_'+train_data.tag+'_{:.3f}_model'.format(accuracy)), step)
                sess.run(train_init_op) 
    train_writer.close()
    sess.close()


if __name__ == '__main__':
    description = 'finetune on other dataset'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset_name', type=str)
    p.add_argument('data_tag', type=str)
    main(**vars(p.parse_args()))
