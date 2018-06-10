from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import logging

import numpy as np
import tensorflow as tf

import i3d
from lib.action_dataset import Action_Dataset
from lib.action_dataset import split_data


_BATCH_SIZE = 6
_CLIP_SIZE = 64
# How many frames are used for each video in testing phase
_EACH_VIDEO_TEST_SIZE = 250
_FRAME_SIZE = 224
_LEARNING_RATE = 0.001
_GLOBAL_EPOCH = 40
_PREFETCH_BUFFER_SIZE = 30
_NUM_PARALLEL_CALLS = 10
_SAVER_MAX_TO_KEEP = 10
_WEIGHT_OF_LOSS_WEIGHT = 7e-7
_MOMENTUM = 0.9
_DROPOUT = 0.36
_OUTPUT_STEP = 20
# When the accuracy on training data higher than this value, run testing phase
_RUN_TEST_THRESH = 0.85
# If the accuracy on testing data higher than this value, save the model
_SAVE_MODEL_THRESH = 0.75
_LOG_ROOT = 'output'

_CHECKPOINT_PATHS = {
    'rgb': './f-data/checkpoints/rgb_scratch/model.ckpt',
    'flow': './f-data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': './f-data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': './f-data/checkpoints/flow_imagenet/model.ckpt',
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


def _get_data_label_from_info(train_info_tensor, name, mode):
    """ Wrapper for `tf.py_func`, get video clip and label from info list."""
    clip_holder, label_holder = tf.py_func(
        process_video, [train_info_tensor, name, mode], [tf.float32, tf.int64])
    return clip_holder, label_holder


def process_video(data_info, name, mode, is_training=True):
    """ Get video clip and label from data info list."""
    data = Action_Dataset(name, mode, [data_info])
    if is_training:
        clip_seq, label_seq = data.next_batch(1, _CLIP_SIZE)
    else:
        clip_seq, label_seq = data.next_batch(
            1, _EACH_VIDEO_TEST_SIZE+1, shuffle=False, data_augment=False)
    clip_seq = 2*(clip_seq/255) - 1
    clip_seq = np.array(clip_seq, dtype='float32')
    return clip_seq, label_seq


def main(dataset='ucf101', mode='rgb', split=1):
    assert mode in ['rgb', 'flow'], 'Only RGB data and flow data is supported'
    log_dir = os.path.join(_LOG_ROOT, 'finetune-%s-%s-%d' %
                           (dataset, mode, split))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, 'log.txt'),
                        filemode='w', format='%(message)s')

    ##  Data Preload  ###
    train_info, test_info = split_data(
        os.path.join('./data', dataset, mode+'.txt'),
        os.path.join('./data', dataset, 'testlist%02d' % split+'.txt'))
#        os.path.join('/data1/yunfeng/i3d_test/data', dataset, mode+'.txt'),
#        os.path.join('/data1/yunfeng/i3d_test/data', dataset, 'testlist%02d' % split+'.txt'))
    train_data = Action_Dataset(dataset, mode, train_info)
    test_data = Action_Dataset(dataset, mode, test_info)

    num_train_sample = len(train_info)
    # Every element in train_info is shown as below:
    # ['v_ApplyEyeMakeup_g08_c01',
    # '/data4/zhouhao/dataset/ucf101/jpegs_256/v_ApplyEyeMakeup_g08_c01',
    # '121', '0']
    train_info_tensor = tf.constant(train_info)
    test_info_tensor = tf.constant(test_info)

    # Dataset building
    # Phase 1 Trainning
    # one element in this dataset is (train_info list)
    train_info_dataset = tf.data.Dataset.from_tensor_slices(
        (train_info_tensor))
    # one element in this dataset is (single image_postprocess, single label)
    # one element in this dataset is (batch image_postprocess, batch label)
    train_info_dataset = train_info_dataset.shuffle(
        buffer_size=num_train_sample)
    train_dataset = train_info_dataset.map(lambda x: _get_data_label_from_info(
        x, dataset, mode), num_parallel_calls=_NUM_PARALLEL_CALLS)
    train_dataset = train_dataset.repeat().batch(_BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=_PREFETCH_BUFFER_SIZE)

    # Phase 2 Testing
    # one element in this dataset is (train_info list)
    test_info_dataset = tf.data.Dataset.from_tensor_slices(
        (test_info_tensor))
    # one element in this dataset is (single image_postprocess, single label)
    test_dataset = test_info_dataset.map(lambda x: _get_data_label_from_info(
        x, dataset, mode), num_parallel_calls=_NUM_PARALLEL_CALLS)
    # one element in this dataset is (batch image_postprocess, batch label)
    test_dataset = test_dataset.batch(1).repeat()
    test_dataset = test_dataset.prefetch(buffer_size=_PREFETCH_BUFFER_SIZE)

    # iterator = dataset.make_one_shot_iterator()
    # clip_holder, label_holder = iterator.get_next()
    iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    clip_holder, label_holder = iterator.get_next()
    clip_holder = tf.squeeze(clip_holder,  [1])
    label_holder = tf.squeeze(label_holder, [1])
    clip_holder.set_shape(
        [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL[mode]])
    dropout_holder = tf.placeholder(tf.float32)
    is_train_holder = tf.placeholder(tf.bool)

    # inference module
    # Inference Module
    with tf.variable_scope(_SCOPE[train_data.mode]):
        # insert i3d model
        model = i3d.InceptionI3d(
            400, spatial_squeeze=True, final_endpoint='Logits')
        # the line below outputs the final results with logits
        # __call__ uses _template, and _template uses _build when defined
        logits, _ = model(clip_holder, is_training=is_train_holder,
                          dropout_keep_prob=dropout_holder)
        logits_dropout = tf.nn.dropout(logits, dropout_holder)
        # To change 400 classes to the ucf101 or hdmb classes
        fc_out = tf.layers.dense(
            logits_dropout, _CLASS_NUM[dataset], use_bias=True)
        # compute the top-k results for the whole batch size
        is_in_top_1_op = tf.nn.in_top_k(fc_out, label_holder, 1)

    # Loss calculation, including L2-norm
    variable_map = {}
    train_var = []
    for variable in tf.global_variables():
        tmp = variable.name.split('/')
        if tmp[0] == _SCOPE[train_data.mode] and 'dense' not in tmp[1]:
            variable_map[variable.name.replace(':0', '')] = variable
        if tmp[-1] == 'w:0' or tmp[-1] == 'kernel:0':
            weight_l2 = tf.nn.l2_loss(variable)
            tf.add_to_collection('weight_l2', weight_l2)
    loss_weight = tf.add_n(tf.get_collection('weight_l2'), 'loss_weight')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_holder, logits=fc_out))
    total_loss = loss + _WEIGHT_OF_LOSS_WEIGHT * loss_weight
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_weight', loss_weight)
    tf.summary.scalar('total_loss', total_loss)

    # Import Pre-trainned model
    saver = tf.train.Saver(var_list=variable_map, reshape=True)
    saver2 = tf.train.Saver(max_to_keep=_SAVER_MAX_TO_KEEP)
    # Specific Hyperparams
    # steps for training: the number of steps on batch per epoch
    per_epoch_step = int(np.ceil(train_data.size/_BATCH_SIZE))
    # global step constant
    global_step = _GLOBAL_EPOCH * per_epoch_step
    # global step counting
    global_index = tf.Variable(0, trainable=False)

    # Set learning rate schedule by hand, also you can use an auto way
    boundaries = [10000, 20000, 30000, 40000, 50000]
    values = [_LEARNING_RATE, 0.0008, 0.0005, 0.0003, 0.0001, 5e-5]
    learning_rate = tf.train.piecewise_constant(
        global_index, boundaries, values)
    tf.summary.scalar('learning_rate', learning_rate)

    # Optimizer set-up
    # FOR BATCH norm, we then use this updata_ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               _MOMENTUM).minimize(total_loss, global_step=global_index)
    sess = tf.Session()
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(train_init_op)
    saver.restore(sess, _CHECKPOINT_PATHS[train_data.mode+'_imagenet'])

    print('----Here we start!----')
    print('Output wirtes to ' + log_dir)
    # logging.info('----Here we start!----')
    step = 0
    # for one epoch
    true_count = 0
    # for 20 batches
    tmp_count = 0
    accuracy_tmp = 0
    epoch_completed = 0
    while step <= global_step:
        step += 1
        start_time = time.time()
        _, loss_now, loss_plus, is_in_top_1, summary = sess.run(
            [optimizer, total_loss, loss_weight, is_in_top_1_op, merged_summary],
            feed_dict={dropout_holder: _DROPOUT, is_train_holder: True})
        duration = time.time() - start_time
        tmp = np.sum(is_in_top_1)
        true_count += tmp
        tmp_count += tmp
        train_writer.add_summary(summary, step)
        # responsible for printing relevant results
        if step % _OUTPUT_STEP == 0:
            accuracy = tmp_count / (_OUTPUT_STEP * _BATCH_SIZE)
            print('step: %-4d, loss: %-.4f, accuracy: %.3f (%.2f sec/batch)' %
                  (step, loss_now, accuracy, float(duration)))
            logging.info('step: % -4d, loss: % -.4f,\
                             accuracy: % .3f ( % .2f sec/batch)' %
                         (step, loss_now, accuracy, float(duration)))
            tmp_count = 0
        if step % per_epoch_step == 0:
            epoch_completed += 1
            accuracy = true_count / (per_epoch_step * _BATCH_SIZE)
            print('Epoch%d, train accuracy: %.3f' %
                  (epoch_completed, accuracy))
            logging.info('Epoch%d, train accuracy: %.3f' %
                         (train_data.epoch_completed, accuracy))
            true_count = 0
            if step % per_epoch_step == 0 and accuracy > _RUN_TEST_THRESH:
                sess.run(test_init_op)
                true_count = 0
                # start test process
                print(test_data.size)
                for i in range(test_data.size):
                    # print(i,true_count)
                    is_in_top_1 = sess.run(is_in_top_1_op,
                                           feed_dict={dropout_holder: 1,
                                                      is_train_holder: False})
                    true_count += np.sum(is_in_top_1)
                accuracy = true_count / test_data.size
                true_count = 0
                # to ensure every test procedure has the same test size
                test_data.index_in_epoch = 0
                print('Epoch%d, test accuracy: %.3f' %
                      (epoch_completed, accuracy))
                logging.info('Epoch%d, test accuracy: %.3f' %
                             (train_data.epoch_completed, accuracy))
                # saving the best params in test set
                if accuracy > _SAVE_MODEL_THRESH:
                    if accuracy > accuracy_tmp:
                        accuracy_tmp = accuracy
                        saver2.save(sess, os.path.join(log_dir,
                                                       test_data.name+'_'+train_data.mode +
                                                       '_{:.3f}_model'.format(accuracy)), step)
                sess.run(train_init_op)
    train_writer.close()
    sess.close()


if __name__ == '__main__':
    description = 'Finetune I3D model on other datasets (such as UCF101 and \
        HMDB51)'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset', type=str, help="name of dataset, e.g., ucf101")
    p.add_argument('mode', type=str, help="type of data, e.g., rgb")
    p.add_argument('split', type=int, help="split of data, e.g., 1")
    main(**vars(p.parse_args()))
