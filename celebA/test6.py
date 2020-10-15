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
from data import flags
from models import get_G, get_D, get_Ea, get_Ec, get_RNN, get_CA
from process import black_male_young_ds, blond_male_young_ds, brown_male_young_ds, black_female_young_ds,\
    blond_female_young_ds, brown_female_young_ds, black_male_old_ds, blond_male_old_ds, brown_male_old_ds, \
    black_female_old_ds, blond_female_old_ds, brown_female_old_ds

os.environ["CUDA_VISIBLE_DEVICES"] = ""
test_epoch = 20
img_size = 128
test_batch_size = 12
row_dim = 12
col_dim = 7


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
    for _, data in enumerate(black_male_young_ds):
        black_male_young_batch = data[0]
    for _, data in enumerate(blond_male_young_ds):
        blond_male_young_batch = data[0]
    for _, data in enumerate(brown_male_young_ds):
        brown_male_young_batch = data[0]
    for _, data in enumerate(black_female_young_ds):
        black_female_young_batch = data[0]
    for _, data in enumerate(blond_female_young_ds):
        blond_female_young_batch = data[0]
    for _, data in enumerate(brown_female_young_ds):
        brown_female_young_batch = data[0]

    for _, data in enumerate(black_male_old_ds):
        black_male_old_batch = data[0]
    for _, data in enumerate(blond_male_old_ds):
        blond_male_old_batch = data[0]
    for _, data in enumerate(brown_male_old_ds):
        brown_male_old_batch = data[0]
    for _, data in enumerate(black_female_old_ds):
        black_female_old_batch = data[0]
    for _, data in enumerate(blond_female_old_ds):
        blond_female_old_batch = data[0]
    for _, data in enumerate(brown_female_old_ds):
        brown_female_old_batch = data[0]

    ###======================== GET SAMPLE ===================================###

    
    sample_image = np.ones([row_dim * col_dim, img_size, img_size, 3], dtype=np.float32)
    for i in range(row_dim):
        for j in range(col_dim):
            sample_image[i * col_dim + j] = blond_female_old_batch[i]

    ###======================== GET SAMPLE2 ===================================###

    sample_image_2 = np.ones([row_dim * col_dim, img_size, img_size, 3], dtype=np.float32)
    for i in range(row_dim):
        for j in range(col_dim):
            sample_image_2[col_dim * i + j] = black_male_young_batch[i]
    print("sample_image_2 shape: ", sample_image_2.shape)

    sample_sentence_tmp = [[0., 0., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0., 0.],
                           [1., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 0., 1.],
                           [0., 1., 0., 0., 0., 0., 1.]] * row_dim
    sample_sentence = tf.constant(sample_sentence_tmp)

    sample_sentence_2_tmp = [[0., 0., 0., 0., 0., 0., 0.], # empty
                             [0., 0., 1., 0., 0., 0., 0.], # brown
                             [1., 0., 0., 0., 0., 0., 0.], # black
                             [0., 0., 0., 0., 1., 0., 0.], # female
                             [0., 0., 0., 0., 0., 0., 0.], # male
                             [0., 0., 0., 0., 0., 0., 1.], # old
                             [0., 1., 0., 0., 1., 0., 0.]] * row_dim  # blond old
    sample_sentence_2 = tf.constant(sample_sentence_2_tmp)

    ###======================== GET SAMPLE4 ===================================###
    sample_image_4 = np.ones([row_dim * col_dim, img_size, img_size, 3], dtype=np.float32)
    for i in range(row_dim):
        for j in range(col_dim):
            sample_image_4[col_dim * i + j] = blond_female_young_batch[i]
    print("sample_image_4: ", sample_image_4.shape)

    sample_sentence_4_tmp_1 = [[1., 0., 0., 0., 0., 0., 0.]] * 14
    sample_sentence_4_tmp_2 = [[0., 1., 0., 0., 0., 0., 0.]] * 14
    sample_sentence_4_tmp_3 = [[0., 0., 0., 1., 0., 0., 0.]] * 14
    sample_sentence_4_tmp_4 = [[0., 0., 0., 0., 1., 0., 0.]] * 14
    sample_sentence_4_tmp_5 = [[0., 0., 0., 0., 0., 0., 1.]] * 14
    sample_sentence_4_tmp_6 = [[0., 0., 0., 0., 0., 0., 0.]] * 14
    sample_sentence_4 = tf.concat([sample_sentence_4_tmp_1,
                                   sample_sentence_4_tmp_2,
                                   sample_sentence_4_tmp_3,
                                   sample_sentence_4_tmp_4,
                                   sample_sentence_4_tmp_5,
                                   sample_sentence_4_tmp_6], 0)
    print("sample_sentence_4: ", sample_sentence_4.shape)

    # for i in range(10):
    #     print(sample_sentence_2[1+i*7])
    # exit()
    tl.visualize.save_images(sample_image, [row_dim, col_dim],
                             flags.sample_dir + '/' + flags.prefix + '/test/examination_test_Xa.png')
    tl.visualize.save_images(sample_image_2, [row_dim, col_dim],
                             flags.sample_dir + '/' + flags.prefix + '/test/examination_test_Xb.png')

    tl.visualize.save_images(sample_image_4, [row_dim, col_dim],
                             flags.sample_dir + '/' + flags.prefix + '/test/examination_test_Xc.png')
    # exit()
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

    # print(sample_image.min())
    # print(sample_image.max())
    # print(sample_image.shape)

    def test_image_1(epoch):
        test_Xa_appearance = Ea(sample_image)
        test_Xa_content = Ec(sample_image)
        print(test_Xa_content.shape)
        print(test_Xa_appearance.shape)
        print(sample_sentence.shape)
        # exit()
        fake_test_Xa = G([test_Xa_appearance, test_Xa_content, sample_sentence])  # change it into female and black hair
        fake_test_Xa = np.array(fake_test_Xa)
        tl.visualize.save_images(fake_test_Xa, [row_dim, col_dim],
                                 '{}/{}/test/examination_test1_train_{:02d}.png'.format(flags.sample_dir, flags.prefix,
                                                                                        epoch))

        # recon_test_Xa = G([test_Xa_appearance, test_Xa_content, match_sentence])
        # recon_test_Xa = np.array(recon_test_Xa)
        # tl.visualize.save_images(recon_test_Xa, [ni, ni],
        #                         '{}/{}/test/examination_test1_recon_{:02d}.png'.format(flags.sample_dir, flags.prefix,
        #                                                                                epoch))

    def test_image_2(epoch):
        test_Xa_content = Ec(sample_image)  # blond female
        test_Xb_appearance = Ea(sample_image_2)  # black male
        fake_test_Xa = G([test_Xb_appearance, test_Xa_content, sample_sentence_2])
        fake_test_Xa = np.array(fake_test_Xa)
        tl.visualize.save_images(fake_test_Xa, [row_dim, col_dim],
                                 '{}/{}/test/examination_test2_train_{:02d}.png'.format(flags.sample_dir, flags.prefix,
                                                                                        epoch))

    def test_image_3(epoch):
        test_Xa_content = Ec(sample_image)
        appearance = np.random.normal(loc=0.0, scale=1.0, size=[col_dim * row_dim, flags.z_dim]).astype(np.float32)
        fake_test_Xa = G([appearance, test_Xa_content, sample_sentence_2])
        fake_test_Xa = np.array(fake_test_Xa)
        tl.visualize.save_images(fake_test_Xa, [row_dim, col_dim],
                                 '{}/{}/test/examination_test3_train_{:02d}.png'.format(flags.sample_dir,
                                                                                        flags.prefix, epoch))

    def test_image_4(epoch):
        test_content = Ec(sample_image_4)
        test_app_1 = Ea(sample_image)
        test_app_2 = Ea(sample_image_2)
        test_app = np.ones([col_dim * row_dim, flags.z_dim], dtype=np.float32)
        for i in range(col_dim):
            alpha = i / (col_dim - 1)
            for j in range(row_dim):
                test_idx = i + j * col_dim
                test_app[test_idx] = alpha * test_app_1[test_idx] + (1 - alpha) * test_app_2[test_idx]
        fake_test_Xa = G([test_app, test_content, sample_sentence_4])
        fake_test_Xa = np.array(fake_test_Xa)
        tl.visualize.save_images(fake_test_Xa, [row_dim, col_dim],
                                 '{}/{}/test/examination_test4_train_{:02d}.png'.format(flags.sample_dir,
                                                                                        flags.prefix, epoch))

    #    G.eval()
    #    Ec.eval()
    #    Ea.eval()
    #    D.eval()

    test_image_1(test_epoch)
    test_image_2(test_epoch)
    test_image_3(test_epoch)
    test_image_4(test_epoch)

    #    G.train()
    #    Ec.train()
    #    Ea.train()
    #    D.train()

    print('[*] end testing')


if __name__ == '__main__':
    test()
