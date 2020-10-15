# -*- coding: utf-8 -*-
"""
@author: Zbc (latest version Aug 15 10:14 2019)
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
hair_color = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]  # hair:0/1/2 black/blond/brown
gender = [[0, 0], [1, 0], [0, 1]]  # [1, 0] -- male [0, 1] -- female
age = [[0, 0], [1, 0], [0, 1]]


# this flags class is temporary
class FLAGS2(object):
    def __init__(self):
        self.batch_size = flags.batch_size  # training batch size
        self.n_epoch = 20 # training epochs
        self.shuffle_buffer_size = 4
        self.test_samples = 25 # sample num for testing

flags2 = FLAGS2()
hair_name = ['black', 'blond', 'brown']
gender_name = ['female', 'male']
age_name = ['young', 'old']

# get_dataset: 
# args: img_size: image width(height)
# return: train set, train set size, test set, test set size
# train (test) set is enumerable tf dataset, each element is a pair (a1,a2), 
# where a1 is batchsize*128*128*3 image tensor with type tf.float32 and range (-1,1), 
# a2 is batch_size*7 one-hot label vector with type tf.float32 and range {0.0, 1.0}
def get_dataset(img_size=128, hair=0, male=1, age_w=0, cnt=100):  # hair:0/1/2 black/blond/brown     male: 0/1 female/male  age_w: 0/1 young/old
    
    # label prepro
    label_file = open('../celebA/list_attr_celeba.txt')
    labels = label_file.readlines()

    keys = labels[1].split()  # name of attr

    attr_keys = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male']

    idx = [8, 9, 11, 20]  # selected labels
    for i in range(4):
        assert keys[idx[i]] == attr_keys[i]

    img_name_list = []
    match_vec_list = []
    tmp_cnt = 0
    for i in range(2, len(labels)):
        l = labels[i].split()[1:]  # split each line
        assert len(l) == len(keys)
        oh = [l[j] for j in idx]
        if oh[0:3] == ['-1', '-1', '-1']:  # skip images without ground-truth hair color label
            continue
        # discard images that have at least 2 hair color labels
        if oh[0] == '1' and oh[1] == '1' or oh[1] == '1' and oh[2] == '1' or oh[0] == '1' and oh[2] == '1':
            continue
        tmp_cnt += 1
        if tmp_cnt <= 100000:
            continue
        name = labels[i].split()[0]
        if hair == 0:  # black hair
            if male == 1:  # male
                if oh[0] == '1' and oh[3] == '1' :
                    img_name_list.append(name)  # add valid img dir
                    match_vec_list.append([1, 0, 0, 1, 0])
            else:  # female
                if oh[0] == '1' and oh[3] == '-1' :
                    img_name_list.append(name)
                    match_vec_list.append([1, 0, 0, 0, 1])

        elif hair == 1:  # blond hair
            if male == 1:  # male
                if oh[1] == '1' and oh[3] == '1' :
                    img_name_list.append(name)  # add valid img dir
                    match_vec_list.append([0, 1, 0, 1, 0])
            else:  # female
                if oh[1] == '1' and oh[3] == '-1' :
                    img_name_list.append(name)
                    match_vec_list.append([0, 1, 0, 0, 1])
        else:  # brown hair
            if male == 1:  # male
                if oh[2] == '1' and oh[3] == '1' :
                    img_name_list.append(name)  # add valid img dir
                    match_vec_list.append([0, 0, 1, 1, 0])
            else:  # female
                if oh[2] == '1' and oh[3] == '-1' :
                    img_name_list.append(name)
                    match_vec_list.append([0, 0, 1, 0, 1])

    img_list = []

    for i in range(len(img_name_list)):
        img_list.append('../celebA/img_align_celeba/' + str(img_name_list[i]))

    test_list = [(i1, i2) for (i1, i2) in zip(img_list, match_vec_list)]
    # print(len(match_vec_list), len(img_list))
    # random.shuffle(test_list)
    test_list = test_list[0:cnt]

    f = open(flags.sample_dir + '/' + flags.prefix + '/test/' + "hair_" + str(hair_name[hair]) + "_gender_" + \
              str(gender_name[male]) + '_test.txt', 'w')
    for i in range(len(test_list)):
         f.write(str(i) + str(test_list[i][0])+'\n')
    f.close()

    # dataset
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
        # image = tf.image.random_flip_left_right(image)  # need random flip ??
        image = image * 2 - 1  # change RGB to -1~1
        # image = image / 127.5 - 1.
        return image

    test_ds = tf.data.Dataset.from_generator(generator_test, output_types=(tf.string, tf.float32))
    # test_ds = test_ds.shuffle(buffer_size=flags2.shuffle_buffer_size)
    test_ds = test_ds.map(lambda x1, x2: (_map_fn(x1), x2), num_parallel_calls=multiprocessing.cpu_count())
    test_ds = test_ds.batch(flags2.test_samples)
    test_ds = test_ds.prefetch(buffer_size=4)
    # print(test_ds)
    return test_ds


black_male_ds = get_dataset(128, 0, 1, 25)
blond_male_ds = get_dataset(128, 1, 1, 25)
brown_male_ds = get_dataset(128, 2, 1, 25)

black_female_ds = get_dataset(128, 0, 0, 25)
blond_female_ds = get_dataset(128, 1, 0, 25)
brown_female_ds = get_dataset(128, 2, 0, 25)


