# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 2019

@original: zgq
 Modification: Zbc, Zyh
"""

import random, os, time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import copy
# from data import train_ds, train_ds_size, im_test, flags
from data_acc import flags
from models import get_G, get_D, get_Ea, get_Ec, get_RNN, get_CA
from datav3_acc import flags2, test_ds

os.environ["CUDA_VISIBLE_DEVICES"] = ""
test_epoch = 39
img_size = 128
test_batch_size = flags2.test_samples
save_path1= 'test_acc'
save_path2= 'test_acc2'

def image_aug_fn_for_test(x):
    x = tl.prepro.imresize(x, [flags.im_sz, flags.im_sz])
    x = x / 127.5 - 1.
    x = x.astype(np.float32)
    return x


def _map_fn(image_path):  # latest version: Zyh 2019/08/14
    # print(image_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize([image], (img_size, img_size))[0]
    image = image * 2 - 1  # change RGB to -1~1
    return image


def _map_fn2(image_path):  # latest version: Zyh 2019/08/14
    # print(image_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = image[20:198, :]  # crop to square
    image = tf.image.resize([image], (img_size, img_size))[0]
    image = image * 2 - 1  # change RGB to -1~1
    return image


def test():
    ni = int(np.ceil(np.sqrt(test_batch_size)))
    print(ni)

    tl.files.exists_or_mkdir(save_path1+'/img_vacant_b', verbose=False)
    tl.files.exists_or_mkdir(save_path1+'/img_vacant_b/black', verbose=False)
    tl.files.exists_or_mkdir(save_path1+'/img_vacant_b/blond', verbose=False)
    tl.files.exists_or_mkdir(save_path1+'/img_vacant_b/brown', verbose=False)
    tl.files.exists_or_mkdir(save_path1+'/img_vacant_b/male', verbose=False)
    tl.files.exists_or_mkdir(save_path1+'/img_vacant_b/female', verbose=False)
    tl.files.exists_or_mkdir(save_path1+'/img_vacant_b/blond_female', verbose=False)
    tl.files.exists_or_mkdir(save_path1+'/img_vacant_b/black_male', verbose=False)

    ###======================== DEFIINE MODEL ===============================###

    # generator
    G = get_G()
    # discriminator
    D = get_D()
    # encoder for appearance
    Ea = get_Ea()
    # encoder for content
    Ec = get_Ec()

    # G.eval()
    # D.eval()
    # Ea.eval()
    # Ec.eval()
    G.train()
    D.train()
    Ea.train()
    Ec.train()

    ###============================ LOAD EXISTING MODELS ====================###
    save_dir_g = os.path.join(flags.model_dir, flags.prefix, 'G_{}.h5'.format(str(test_epoch)))
    save_dir_d = os.path.join(flags.model_dir, flags.prefix, 'D_{}.h5'.format(str(test_epoch)))
    save_dir_ea = os.path.join(flags.model_dir, flags.prefix, 'Ea_{}.h5'.format(str(test_epoch)))
    save_dir_ec = os.path.join(flags.model_dir, flags.prefix, 'Ec_{}.h5'.format(str(test_epoch)))
    # save_dir_ca = os.path.join(flags.model_dir, flags.prefix, 'CA_{}.h5'.format(str(test_epoch)))
    if G.load_weights(save_dir_g) is False:
        raise Exception("missing G model")
    if D.load_weights(save_dir_d) is False:
        raise Exception("missing D model")
    if Ea.load_weights(save_dir_ea) is False:
        raise Exception("missing Ea model")
    if Ec.load_weights(save_dir_ec) is False:
        raise Exception("missing Ec model")

    ###============================ TESTING ================================###
    print('[*] start testing')

    change_label = np.load(save_path1 + '/Xa_r_label_vacant.npy')
    change_label = change_label.astype(np.float32)
    X_path = np.load(save_path1+'/Xa_vacant.npy')
    X_num = change_label.shape[0]
    X_img = np.ones([X_num, img_size, img_size, 3], dtype=np.float32)
    change_name_list = []

    X_ans_label = np.load(save_path1+'/Xa_ans_label_vacant.npy')
    X_ans_label = X_ans_label.astype(np.float32)

    for i in range(X_num):
        X_img[i] = _map_fn2('/raid/zhaoyihao/celebA/img_align_celeba/'+str(X_path[i]))
    print("test_num: ", X_num)

    epoch = int(X_num / test_batch_size)
    for i in range(epoch):
        X_label = change_label[i*test_batch_size : (i+1)*test_batch_size]
        X_pr_label = X_ans_label[i*test_batch_size : (i+1)*test_batch_size]
        X_test = X_img[i*test_batch_size : (i+1)*test_batch_size]
        test_X_a = Ea(X_test)
        test_X_c = Ec(X_test)
        fake_X = G([test_X_a, test_X_c, X_label])
        fake_X = np.array(fake_X)
        for j in range(test_batch_size):
            X_idx = i*test_batch_size+j
            tl.visualize.save_images(fake_X[j:j+1], [1,1], save_path1 + '/img_vacant_b/changed_'+str(X_path[X_idx]))
            if list(X_pr_label[j][0:3])==[1,0,0]:
                tl.visualize.save_images(fake_X[j:j+1], [1,1], save_path1 + '/img_vacant_b/black/changed_'+str(X_path[X_idx]))
            elif list(X_pr_label[j][0:3])==[0,1,0]:
                tl.visualize.save_images(fake_X[j:j+1], [1,1], save_path1 + '/img_vacant_b/blond/changed_'+str(X_path[X_idx]))
            elif list(X_pr_label[j][0:3])==[0,0,1]:
                tl.visualize.save_images(fake_X[j:j+1], [1,1], save_path1 + '/img_vacant_b/brown/changed_'+str(X_path[X_idx]))
            if list(X_pr_label[j][3:5])==[1,0]:
                tl.visualize.save_images(fake_X[j:j+1], [1,1], save_path1 + '/img_vacant_b/male/changed_'+str(X_path[X_idx]))
            elif list(X_pr_label[j][3:5])==[0,1]:
                tl.visualize.save_images(fake_X[j:j+1], [1,1], save_path1 + '/img_vacant_b/female/changed_'+str(X_path[X_idx]))
            if list(X_pr_label[j])==[1,0,0,1,0]:
                tl.visualize.save_images(fake_X[j:j+1], [1,1], save_path1 + '/img_vacant_b/black_male/changed_'+str(X_path[X_idx]))
            if list(X_pr_label[j])==[0,1,0,0,1]:
                tl.visualize.save_images(fake_X[j:j+1], [1,1], save_path1 + '/img_vacant_b/blond_female/changed_'+str(X_path[X_idx]))
            change_name_list.append('changed_'+str(X_path[X_idx]))

    change_name_arr = np.array(change_name_list)
    print(change_name_arr.shape)
    np.save(save_path1 + '/Xa_pr_vacant_b.npy', change_name_arr)
#    np.save('test_acc/Xa_label.npy', X_label)

    print('[*] end testing')


if __name__ == '__main__':
    test()
