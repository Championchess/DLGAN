# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:03:04 2019

@author: Yhq, Zbc, Zyh, Zgq (latest version Aug 27 14:14 2019)
image, match_vec, relevant_match_vec, relevant_vec, ground_truth_vec, relevant_gt_vec
"""

# update: test set be a tensor

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os, re
import random
import multiprocessing
import copy
from data import flags

# dictionaries
hair_color = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
gender = [[0, 0], [1, 0], [0, 1]]  # [1, 0] -- y [0, 1] -- n
age = []

#permit vacant label for 2 label case
def random_vec_2():
    tmp_vec = []
    idx_tmp1 = np.random.choice(3, 1)
    idx_tmp2 = np.random.choice(2, 1)
    tmp_vec += hair_color[idx_tmp1[0]+1]
    tmp_vec += gender[idx_tmp2[0]+1]
    return tmp_vec  
    
def get_any_vec_2(match_vec):
    any_vec = random_vec_2()
    #while any_vec[0:3] != match_vec[0:3] and any_vec[3:5] != match_vec[3:5]:
        #any_vec = random_vec()
    return any_vec


def get_mis_vec_2(match_vec):  # 000/00
    mis_vec = random_vec_2()
    while mis_vec == match_vec or (mis_vec[0:3] == [0, 0, 0] and mis_vec[3:5] == match_vec[3:5]) or (mis_vec[0:3] == match_vec[0:3] and mis_vec[3:5] == [0, 0]) or mis_vec == [0, 0, 0, 0, 0]:
        mis_vec = random_vec_2()
    return mis_vec  


def get_disable_match(match_vec):
    disable_match = copy.deepcopy(match_vec)
    idx = np.random.choice(4, 1)
    if idx[0] == 0:  # 1/4 prob to return ground-truth vec
        return disable_match
    elif idx[0] == 1:  # 1/4 prob to change the hair attr
        disable_match[0:3] = [0, 0, 0]
    elif idx[0] == 2:  # 1/4 prob to change the gender attr
        disable_match[3:5] = [0, 0]
    else:   # 1/4 prob to return fully-vacant vec
        disable_match = [0, 0, 0, 0, 0]
    return disable_match


# this flags class is temporary
class FLAGS2(object):
    def __init__(self):
        self.batch_size = flags.batch_size  # training batch size
        self.n_epoch = 200 # training epochs
        self.shuffle_buffer_size = 128
        self.test_samples = 81 # sample num for testing

flags2 = FLAGS2()


# get_dataset: 
# args: img_size: image width(height)
# return: train set, train set size, test set, test set size
# train (test) set is enumerable tf dataset, each element is a pair (a1,a2), 
# where a1 is batchsize*128*128*3 image tensor with type tf.float32 and range (-1,1), 
# a2 is batch_size*7 one-hot label vector with type tf.float32 and range {0.0, 1.0}
def get_dataset(img_size=flags.im_sz):
    
    # label prepro
    label_file = open('/raid/zhaoyihao/celebA/list_attr_celeba.txt')
    labels = label_file.readlines()

    keys = labels[1].split()  # name of attr
    # print(len(keys)) # 40

    attr_keys = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male']
    idx = [8, 9, 11, 20]  # selected labels
    for i in range(4):
        assert keys[idx[i]] == attr_keys[i]

    # construct the matching text dataset
    match_vec_list = []
    r_m_vec_list = []
    any_vec_list = []
    ground_truth_list = []
    r_gt_list = []
    count = 0  # count pics we don't need
    double_cnt = 0
    img_name_list = []
    for i in range(2, len(labels)):
        match_vec = []
        l = labels[i].split()[1:]  # split each line
        assert len(l) == len(keys)
        oh = [l[j] for j in idx]
        if oh[0:3] == ['-1', '-1', '-1']:  # skip images without ground-truth hair color label
            count += 1
            continue
        # discard images that have at least 2 hair color labels
        if oh[0] == '1' and oh[1] == '1' or oh[1] == '1' and oh[2] == '1' or oh[0] == '1' and oh[2] == '1':
            double_cnt += 1
            continue

        if oh[0] == '1':
            match_vec += hair_color[1]
        else:
            if oh[1] == '1':
                match_vec += hair_color[2]
            else:
                match_vec += hair_color[3]

        # fetch gender attr
        if oh[3] == '1':
            match_vec += gender[1]
        else:
            match_vec += gender[2]

        img_name_list.append(labels[i].split()[0])  # add valid img dir

#        mis_vec = get_mis_vec_2(match_vec)
#        mis_vec_list.append(mis_vec)

        any_vec = get_any_vec_2(match_vec)
        any_vec_list.append(any_vec)

        empty_front = 0
        empty_back = 0
        if any_vec[0:3]==[0, 0, 0] :
            empty_front = 1
        if any_vec[3:5]==[0, 0] :
            empty_back = 1
            
        if empty_front and empty_back :
            r_m_vec = [0, 0, 0, 0, 0]
        elif (not empty_front) and empty_back :
            r_m_vec = match_vec[0:3] + [0, 0]
        elif empty_front and (not empty_back) :
            r_m_vec = [0, 0, 0] + match_vec[3:5]
        else :
            r_m_vec = match_vec[:]
        r_m_vec_list.append(r_m_vec)
        
        if empty_front and empty_back :
            r_gt_vec = match_vec
        elif (not empty_front) and empty_back :
            r_gt_vec = any_vec[0:3] + match_vec[3:5]
        elif empty_front and (not empty_back) :
            r_gt_vec = match_vec[0:3] + any_vec[3:5]
        else :
            r_gt_vec = any_vec
        r_gt_list.append(r_gt_vec)
        
        ground_truth_list.append(match_vec)
        
        disable_match_vec = get_disable_match(match_vec)
        match_vec_list.append(disable_match_vec)
    assert len(match_vec_list) == len(labels) - 2 - count - double_cnt
    # combine data, split train/test set
    # img_list = tl.files.load_file_list(path='celebA/img_align_celeba/', regx='.*.jpg', keep_prefix=True, printable=False)
    # img_list.sort(key=lambda x: int(re.match('\D+(\d+)\.jpg', x).group(1)))  # rank by num

    img_list = []
    for i in range(len(img_name_list)):
        img_list.append('/raid/zhaoyihao/celebA/img_align_celeba/'+ str(img_name_list[i]))
    # print(img_list)
    assert len(img_list) == len(labels) - 2 - count - double_cnt
    assert len(img_list) == len(match_vec_list)
    assert len(img_list) == len(r_m_vec_list)
    assert len(img_list) == len(any_vec_list)
    assert len(img_list) == len(ground_truth_list)
    assert len(img_list) == len(r_gt_list)

    # image -- label(vector)
    data_list = [(i1, i2, i3, i4, i5, i6) for (i1, i2, i3, i4, i5, i6) in zip(img_list, match_vec_list, r_m_vec_list, any_vec_list, ground_truth_list, r_gt_list)]
    # print(len(data_list), data_list[0])

    # data set partition rule is in list_eval_partition.txt !
    test_list = data_list[100000:]
    train_list = data_list[0:100000]
    #print(test_list[0][0])
    #exit()
    print('train set size: {} test set size: {}'.format(len(train_list), len(test_list)))
#    random.shuffle(test_list)
    random.shuffle(train_list)
    
    # add list for B, it is a deep copy !
    train_list_B = train_list[:]
    random.shuffle(train_list_B)
    # print(train_list[0], train_list_B[0])
    # exit()
    '''
    # manually check the correspondence
    for i in data_list:
        print(i)
    '''
    
    # dataset
    def generator_train():
        for comb in train_list:
            yield comb

    def generator_train_B():
        for comb in train_list_B:
            yield comb

    def generator_test():
        for comb in test_list:
            yield comb

    def _map_fn(image_path):  # latest version: Zyh 2019/08/14
        # print(image_path)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = image[20:198, :]  # crop to square
        image = tf.image.resize([image], (img_size, img_size))[0]
        # image = tl.prepro.imresize(image, [img_size, img_size])
        image = tf.image.random_flip_left_right(image)  # need random flip ??
        image = image * 2 - 1  # change RGB to -1~1
        # image = image / 127.5 - 1.
        return image
    
    train_ds_A = tf.data.Dataset.from_generator(generator_train, output_types=(tf.string, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
    train_ds_A = train_ds_A.shuffle(buffer_size=flags2.shuffle_buffer_size)
    train_ds_A = train_ds_A.map(lambda x1, x2, x3, x4, x5, x6: (_map_fn(x1), x2, x3, x4, x5, x6), num_parallel_calls=multiprocessing.cpu_count())
    train_ds_A = train_ds_A.batch(flags2.batch_size)
    train_ds_A = train_ds_A.prefetch(buffer_size=4)
    
    train_ds_B = tf.data.Dataset.from_generator(generator_train_B, output_types=(tf.string, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
    train_ds_B = train_ds_B.shuffle(buffer_size=flags2.shuffle_buffer_size)
    train_ds_B = train_ds_B.map(lambda x1, x2, x3, x4, x5, x6: (_map_fn(x1), x2, x3, x4, x5, x6), num_parallel_calls=multiprocessing.cpu_count())
    train_ds_B = train_ds_B.batch(flags2.batch_size)
    train_ds_B = train_ds_B.prefetch(buffer_size=4)
    
    test_ds = tf.data.Dataset.from_generator(generator_test, output_types=(tf.string, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
#    test_ds = test_ds.shuffle(buffer_size=flags2.shuffle_buffer_size)
    test_ds = test_ds.map(lambda x1, x2, x3, x4, x5, x6: (_map_fn(x1), x2, x3, x4, x5, x6), num_parallel_calls=multiprocessing.cpu_count())
    test_ds = test_ds.batch(flags2.test_samples)
    test_ds = test_ds.prefetch(buffer_size=4)
    
    return train_ds_A, train_ds_B, len(train_list), test_ds, len(test_list)
    

# from test dataset generate 16 samples
# args: test_dataset: test dataset generate by get_dataset function
# return: two test samples, each is a pair (a1,a2),
# where a1 is 16*128*128*3 image tensor with type tf.float32 and range (-1,1), 
# a2 is 16*7 one-hot label vector with type tf.float32 and range {0.0, 1.0}
def pick_test_samples(test_dataset):
    i = 0
    for s in test_dataset:
        if i == 0:
            sample1 = s
            i += 1
        elif i == 1:
            sample2 = s
            i += 1
        else:
            return sample1, sample2
    
    
train_ds_A, train_ds_B, train_size, test_ds, test_size = get_dataset()
test_samples_1, test_samples_2 = pick_test_samples(test_ds)
#print(train_ds_A, train_ds_B, train_size, test_ds, test_size, all_text)
#print(test_samples_1, test_samples_2)
