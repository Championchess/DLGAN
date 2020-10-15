# -*- coding: utf-8 -*-
"""
Created on Sat 20/7/2019

@original: zgq

Latest version: zyh/zgq ITGAN main pipeline with starGAN label loss 25/8/2019
"""
import random, os, time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import re, nltk, string
import copy
# from data import train_ds, train_ds_size, im_test, flags
from data import flags
from datav3_multi import train_ds_A, train_ds_B, train_size, test_ds, test_size, test_samples_1, test_samples_2
from models import get_G, get_D, get_Ea, get_Ec

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# tf.compat.v1.disable_eager_execution()
start_epoch = 0
start_step = 0
img_size = 128
test_batch_size = 25


def KL_loss(mu, log_sigma):
    loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(2 * log_sigma) - 1 - 2 * log_sigma)
    loss = tf.reduce_mean(loss)
    loss = tf.reduce_mean(loss)
    return loss


def image_aug_fn(x):
    x = tl.prepro.imresize(x, [flags.im_sz, flags.im_sz])

    # M_rotate = tl.prepro.affine_rotation_matrix(angle=(-16, 16))
    M_flip = tl.prepro.affine_horizontal_flip_matrix(prob=0.5)
    # M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.8, 1.2))
    h, w, _ = x.shape
    # M_combined = M_zoom.dot(M_flip).dot(M_rotate)
    M_combined = M_flip
    transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=w, y=h)
    x = tl.prepro.affine_transform_cv2(x, transform_matrix, border_mode='replicate')
    # x = tl.prepro.flip_axis(x, axis=1, is_random=True)
    # x = tl.prepro.rotation(x, rg=16, is_random=True, fill_mode='nearest')
    # x = tl.prepro.imresize(x, size=[int(h * 1.2), int(w * 1.2)], interp='bicubic', mode=None)
    x = tl.prepro.crop(x, wrg=flags.im_sz, hrg=flags.im_sz, is_random=True)
    # x = x / 127.5 - 1.
    x = x.astype(np.float32)
    # print(type(x))
    return x


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
    # image = image[20:198, :]  # crop to square
    image = tf.image.resize([image], (img_size, img_size))[0]
    # image = tl.prepro.imresize(image, [img_size, img_size])
    image = image * 2 - 1  # change RGB to -1~1
    # image = image / 127.5 - 1.
    return image


def _map_fn2(image_path):  # latest version: Zyh 2019/08/14
    # print(image_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = image[20:198, :]  # crop to square
    image = tf.image.resize([image], (img_size, img_size))[0]
    # image = tl.prepro.imresize(image, [img_size, img_size])
    image = image * 2 - 1  # change RGB to -1~1
    # image = image / 127.5 - 1.
    return image


def train():
    ni = int(np.ceil(np.sqrt(test_batch_size)))
    print(ni)

    ###======================== GET SAMPLE(to change it from blonde male into black male) ===================================###
    image_path = np.array(['../img_test/star_1.jpg',
                           '../img_test/star_10001.jpg',
                           '../img_test/star_01010.jpg',
                           '../img_test/star_01010(2).jpg',
                           '../img_test/star_01001.jpg',
                           '../img_test/202254.jpg',
                           '../img_test/202331.jpg',
                           '../img_test/202339.jpg',
                           '../img_test/202377.jpg',
                           '../img_test/202468.jpg',
                           '../img_test/202469.jpg',
                           '../img_test/202510.jpg',
                           '../img_test/202512.jpg',
                           '../img_test/202547.jpg',
                           '../img_test/202560.jpg',
                           '../img_test/202596.jpg',
                           '../img_test/202093.jpg',
                           '../img_test/202068.jpg',
                           '../img_test/202049.jpg',
                           '../img_test/202024.jpg',
                           '../img_test/202000.jpg',
                           '../img_test/202122.jpg',
                           '../img_test/201834.jpg',
                           '../img_test/201830.jpg',
                           '../img_test/201716.jpg'])
    sample_image = np.ones([test_batch_size, img_size, img_size, 3], dtype=np.float32)
    for i in range(0, test_batch_size):
        if i < 5:
            sample_image[i] = _map_fn(image_path[i])
        else:
            sample_image[i] = _map_fn2(image_path[i])
    print(sample_image.shape)

    #    sample_image = test_samples_1[0]
    #    sample_image = tl.prepro.threading_data(sample_image, image_aug_fn_for_test)
    sample_sentence = test_samples_1[3]  # any text
    match_sentence = test_samples_1[1]  # match text

    # sample_image_2 = test_samples_2[0]
    # sample_image_2 = tl.prepro.threading_data(sample_image_2, image_aug_fn_for_test)

    image_path_2 = np.array(['../img_test/black_female/1.jpg',
                             '../img_test/blond_female/1.jpg',
                             '../img_test/black_male/1.jpg',
                             '../img_test/black_male/2.jpg',
                             '../img_test/blond_female/2.jpg',
                             '../img_test/black_female/2.jpg',
                             '../img_test/blond_male/1.jpg',
                             '../img_test/black_female/3.jpg',
                             '../img_test/black_female/4.jpg',
                             '../img_test/black_male/3.jpg',
                             '../img_test/black_female/5.jpg',
                             '../img_test/black_female/6.jpg',
                             '../img_test/black_male/4.jpg',
                             '../img_test/black_female/7.jpg',
                             '../img_test/blond_female/3.jpg',
                             '../img_test/black_male/5.jpg',
                             '../img_test/black_female/8.jpg',
                             '../img_test/black_female/9.jpg',
                             '../img_test/black_female/10.jpg',
                             '../img_test/black_female/11.jpg',
                             '../img_test/black_female/12.jpg',
                             '../img_test/black_female/13.jpg',
                             '../img_test/black_female/14.jpg',
                             '../img_test/black_female/15.jpg',
                             '../img_test/black_female/16.jpg'])

    sample_image_2 = np.ones([test_batch_size, img_size, img_size, 3], dtype=np.float32)
    for i in range(0, test_batch_size):
        sample_image_2[i] = _map_fn2(image_path_2[i])

    image_path_3 = np.array(['../img_test/blond_female/1.jpg',
                             '../img_test/black_female/1.jpg',
                             '../img_test/blond_male/1.jpg',
                             '../img_test/blond_male/2.jpg',
                             '../img_test/black_female/2.jpg',
                             '../img_test/blond_female/2.jpg',
                             '../img_test/black_male/1.jpg',
                             '../img_test/blond_female/3.jpg',
                             '../img_test/blond_female/4.jpg',
                             '../img_test/blond_male/3.jpg',
                             '../img_test/blond_female/5.jpg',
                             '../img_test/blond_female/6.jpg',
                             '../img_test/blond_male/4.jpg',
                             '../img_test/blond_female/7.jpg',
                             '../img_test/black_female/3.jpg',
                             '../img_test/blond_male/5.jpg',
                             '../img_test/blond_female/8.jpg',
                             '../img_test/blond_female/9.jpg',
                             '../img_test/blond_female/10.jpg',
                             '../img_test/blond_female/11.jpg',
                             '../img_test/blond_female/12.jpg',
                             '../img_test/blond_female/13.jpg',
                             '../img_test/blond_female/14.jpg',
                             '../img_test/blond_female/15.jpg',
                             '../img_test/blond_female/16.jpg'])

    sample_image_3 = np.ones([test_batch_size, img_size, img_size, 3], dtype=np.float32)
    for i in range(0, test_batch_size):
        sample_image_3[i] = _map_fn2(image_path_3[i])

    #    sample_image = fix_test_samples_1[0]
    #    sample_image = tl.prepro.threading_data(sample_image, image_aug_fn_for_test)
    # match_sentence = fix_test_samples_1[1]  # match text
    #    match_sentence = tf.constant([[0., 1., 0., 0., 1.]], tf.float32)
    match_sentence = tf.constant([[0., 0., 1., 0., 1.], [1., 0., 0., 0., 1.],
                                  [0., 0., 1., 1., 0.], [0., 0., 1., 1., 0.],
                                  [0., 0., 1., 0., 1.], [0., 1., 0., 0., 1.],
                                  [0., 0., 1., 1., 0.], [0., 1., 0., 0., 1.],
                                  [0., 0., 1., 0., 1.], [0., 0., 1., 1., 0.],
                                  [0., 1., 0., 0., 1.], [0., 1., 0., 0., 1.],
                                  [0., 1., 0., 1., 0.], [0., 1., 0., 0., 1.],
                                  [0., 0., 1., 0., 1.], [0., 1., 0., 1., 0.],
                                  [0., 1., 0., 0., 1.], [0., 1., 0., 0., 1.],
                                  [0., 1., 0., 0., 1.], [0., 1., 0., 0., 1.],
                                  [0., 1., 0., 0., 1.], [0., 1., 0., 0., 1.],
                                  [0., 1., 0., 0., 1.], [0., 1., 0., 0., 1.],
                                  [0., 1., 0., 0., 1.]])
    recon_sentence = tf.zeros_like(match_sentence)
    # sample_sentence_per_1 = tf.constant([[1., 0., 0., 0., 0.]], tf.float32)
    # # sample_sentence_per_1 = tf.constant([[0., 0., 0., 1. , 0.]], tf.float32)
    # # sample_sentence_per_1 = tf.constant([[0., 0., 0., 0. , 0.]], tf.float32)
    # sample_sentence = tf.tile(sample_sentence_per_1, [test_batch_size, 1])
    sample_sentence = tf.constant([[1., 0., 0., 1., 0.], [1., 0., 0., 1., 0.],
                                   [1., 0., 0., 0., 1.], [1., 0., 0., 0., 1.],
                                   [1., 0., 0., 1., 0.], [0., 1., 0., 1., 0.],
                                   [0., 1., 0., 0., 1.], [0., 1., 0., 1., 0.],
                                   [0., 1., 0., 1., 0.], [0., 1., 0., 0., 1.],
                                   [0., 0., 1., 1., 0.], [0., 0., 1., 1., 0.],
                                   [0., 0., 1., 0., 1.], [0., 0., 1., 1., 0.],
                                   [0., 0., 1., 1., 0.], [1., 0., 0., 0., 1.],
                                   [1., 0., 0., 1., 0.], [1., 0., 0., 1., 0.],
                                   [1., 0., 0., 1., 0.], [1., 0., 0., 1., 0.],
                                   [0., 0., 1., 1., 0.], [0., 0., 1., 1., 0.],
                                   [0., 0., 1., 1., 0.], [0., 0., 1., 1., 0.],
                                   [0., 0., 1., 1., 0.]])

    sample_sentence_2 = tf.constant([[0., 0., 0., 1., 0.], [0., 0., 0., 1., 0.],
                                     [0., 0., 0., 0., 1.], [0., 0., 0., 0., 1.],
                                     [0., 0., 0., 1., 0.], [0., 0., 0., 1., 0.],
                                     [0., 0., 0., 0., 1.], [0., 0., 0., 1., 0.],
                                     [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.],
                                     [0., 0., 0., 1., 0.], [0., 0., 0., 1., 0.],
                                     [0., 1., 0., 0., 0.], [0., 1., 0., 0., 0.],
                                     [0., 1., 0., 0., 0.], [0., 1., 0., 0., 0.],
                                     [0., 1., 0., 0., 0.], [0., 1., 0., 0., 0.],
                                     [0., 1., 0., 0., 0.], [0., 1., 0., 0., 0.],
                                     [1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.],
                                     [1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.],
                                     [1., 0., 0., 0., 0.]])
    #for app_4, we use sample_sentence_3
    sample_sentence_3 = tf.constant([[0., 0., 0., 1., 0.], [0., 0., 0., 1., 0.],
                                     [0., 0., 0., 0., 1.], [0., 0., 0., 0., 1.],
                                     [0., 0., 0., 1., 0.], [0., 0., 0., 1., 0.],
                                     [0., 0., 0., 0., 1.], [0., 0., 0., 1., 0.],
                                     [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.],
                                     [0., 0., 0., 1., 0.], [0., 0., 0., 1., 0.],
                                     [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.],
                                     [0., 0., 0., 0., 1.], [0., 0., 0., 1., 0.],
                                     [0., 0., 0., 0., 1.], [0., 0., 0., 0., 1.],
                                     [0., 0., 0., 0., 1.], [0., 0., 0., 0., 1.],
                                     [0., 0., 0., 0., 1.], [0., 0., 0., 0., 1.],
                                     [0., 0., 0., 0., 1.], [0., 0., 0., 0., 1.],
                                     [0., 0., 0., 0., 1.]])

    print("Following are sample sentence:")
    print(sample_sentence)

    print("Following are sample sentence 2:")
    print(sample_sentence_2)

    print("Following are sample sentence 3:")
    print(sample_sentence_3)
    # idexs = tl.utils.get_random_int(min_v=0, max_v=test_size - 1, number=flags.batch_size)

    tl.visualize.save_images(sample_image, [ni, ni], flags.sample_dir + '/' + flags.prefix + '/examination_test_Xa.png')
    tl.visualize.save_images(sample_image_2, [ni, ni],
                             flags.sample_dir + '/' + flags.prefix + '/examination_test_Xb.png')
    tl.visualize.save_images(sample_image_3, [ni, ni],
                             flags.sample_dir + '/' + flags.prefix + '/examination_test_Xc.png')

    '''
    save_dir       = "samples/20190718"
    checkpoint_dir = "checkpoint/20190718"
#    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir(checkpoint_dir)
    '''

    ###======================== DEFIINE MODEL ===============================###
    # RNN = get_RNN()
    # RNN.eval()
    # generator
    G = get_G()
    # discriminator
    D = get_D()
    # D2 = get_Ds(patch_size=(140, 140)) # you may use multiple D for different scales
    # encoder for appearance
    Ea = get_Ea()
    # encoder for content
    Ec = get_Ec()

    # CA = get_CA()  # conditional augmentation for text embedding (can be seem as an apart of G)

    G.train()
    D.train()
    Ea.train()
    Ec.train()
    # CA.train()

    ####======================== DEFINE TRAIN OPTS ==========================###
    lr_v = tf.Variable(flags.lr_init)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=flags.beta_1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=flags.beta_1)
    # ea_optimizer = tf.optimizers.Adam(lr_v, beta_1=flags.beta_1)
    # ec_optimizer = tf.optimizers.Adam(lr_v, beta_1=flags.beta_1)

    n_step_epoch = int(train_size / flags.batch_size)
    # print(train_size)
    # print(flags.batch_size)

    ###============================ LOAD EXISTING MODELS ====================###
    ## load the latest checkpoints
    save_dir_txt = os.path.join(flags.model_dir, '20190724', 'RNN.h5')

    # load pretrained txt
    # if RNN.load_weights(save_dir_txt) is False:
    # raise Exception("missing txt model")

    if start_epoch != 0:

        save_dir_g = os.path.join(flags.model_dir, flags.prefix, 'G_{}.h5'.format(str(start_epoch - 1)))
        save_dir_d = os.path.join(flags.model_dir, flags.prefix, 'D_{}.h5'.format(str(start_epoch - 1)))
        save_dir_ea = os.path.join(flags.model_dir, flags.prefix, 'Ea_{}.h5'.format(str(start_epoch - 1)))
        save_dir_ec = os.path.join(flags.model_dir, flags.prefix, 'Ec_{}.h5'.format(str(start_epoch - 1)))
        # save_dir_ca = os.path.join(flags.model_dir, flags.prefix, 'CA_{}.h5'.format(str(start_epoch-1)))
        if G.load_weights(save_dir_g) is False:
            raise Exception("missing G model")
        if D.load_weights(save_dir_d) is False:
            raise Exception("missing D model")
        if Ea.load_weights(save_dir_ea) is False:
            raise Exception("missing Ea model")
        if Ec.load_weights(save_dir_ec) is False:
            raise Exception("missing Ec model")
        # if CA.load_weights(save_dir_ca) is False:
        # raise Exception("missing CA model")

    ###============================ TRAINING ================================###
    print('[*] start training')

    def test_image_1(epoch, step):
        test_Xa_appearance = Ea(sample_image)
        test_Xa_content = Ec(sample_image)
        # phi_ta_r = RNN(sample_sentence)
        # phi_test_ta_r_aug, m_fake_test_Xa, s_fake_test__Xa = CA(phi_ta_r, disable_aug=0)
        # print(test_Xa_appearance.shape, test_Xa_content.shape, phi_test_ta_r_aug.shape)
        fake_test_Xa = G([test_Xa_appearance, test_Xa_content, sample_sentence])
        fake_test_Xa = np.array(fake_test_Xa)
        # print(str(step)+" sample_sentences:")
        # print(sample_sentence)
        tl.visualize.save_images(fake_test_Xa, [ni, ni],
                                 '{}/{}/examination_test1_train_{:02d}_{:02d}.png'.format(flags.sample_dir,
                                                                                          flags.prefix, epoch, step))

        recon_test_Xa = G([test_Xa_appearance, test_Xa_content, match_sentence])
        recon_test_Xa = np.array(recon_test_Xa)
        tl.visualize.save_images(recon_test_Xa, [ni, ni],
                                 '{}/{}/examination_test1_recon_{:02d}_{:02d}.png'.format(flags.sample_dir,
                                                                                          flags.prefix, epoch, step))

    def test_image_2(epoch, step):
        test_Xa_content = Ec(sample_image)
        test_Xb_appearance = Ea(sample_image_2)
        # phi_ta_r = RNN(sample_sentence)
        # phi_test_ta_r_aug, m_fake_test_Xa, s_fake_test__Xa = CA(phi_ta_r, disable_aug=0)
        # print(test_Xb_appearance.shape)
        # print(test_Xa_content.shape)
        # print(sample_sentence_2.shape)
        fake_test_Xa = G([test_Xb_appearance, test_Xa_content, sample_sentence_2])
        fake_test_Xa = np.array(fake_test_Xa)
        # print(str(epoch)+" sample_sentences:")
        # print(sample_sentence)
        tl.visualize.save_images(fake_test_Xa, [ni, ni],
                                 '{}/{}/examination_test2_train_{:02d}_{:02d}.png'.format(flags.sample_dir,
                                                                                          flags.prefix,
                                                                                          epoch, step))

    def test_image_3(epoch, step, i):
        test_Xa_content = Ec(sample_image)
        # print(test_Xa_content.shape)
        appearance = np.random.normal(loc=0.0, scale=1.0, size=[test_batch_size, flags.z_dim]).astype(np.float32)
        # print(sample_sentence.shape)

        # phi_ta_r = RNN(sample_sentence)
        # phi_test_ta_r_aug, m_fake_test_Xa, s_fake_test__Xa = CA(phi_ta_r, disable_aug=0)
        fake_test_Xa = G([appearance, test_Xa_content, sample_sentence_2])
        fake_test_Xa = np.array(fake_test_Xa)
        # print(str(epoch)+" sample_sentences:")
        # print(sample_sentence)
        tl.visualize.save_images(fake_test_Xa, [ni, ni],
                                 '{}/{}/examination_test3_train_{:02d}_{:02d}_{:02d}.png'.format(flags.sample_dir,
                                                                                                 flags.prefix, epoch,
                                                                                                 step, i))

    def test_image_4(epoch, step, alpha):
        test_content = Ec(sample_image)
        test_app_1 = Ea(sample_image_2)
        test_app_2 = Ea(sample_image_3)
        test_app = alpha * test_app_1 + (1 - alpha) * test_app_2
        fake_test_Xa = G([test_app, test_content, sample_sentence_2])
        fake_test_Xa = np.array(fake_test_Xa)
        # print(str(epoch)+" sample_sentences:")
        # print(sample_sentence)
        tl.visualize.save_images(fake_test_Xa, [ni, ni],
                                 '{}/{}/examination_test4_train_{:02d}_{:02d}_{:02f}.png'.format(flags.sample_dir,
                                                                                                 flags.prefix, epoch,
                                                                                                 step, alpha))

    def pre_test():
        test_image_1(-1, -1)
        test_image_2(-1)
        for i in range(10):
            test_image_3(-1, i)

    def gradient_penalty(f, real, fake):  # parameters: D, real_image, fake_image
        def _interpolate(a, b):
            alpha = tf.random.uniform(shape=[a.shape[0], 1, 1, 1], minval=0., maxval=1.)
            inter = alpha * a + (1 - alpha) * b
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred, _ = f(x)
        grad = t.gradient(pred, x)
        grad = tf.reshape(grad, [tf.shape(grad)[0], -1])
        norm = tf.math.sqrt(1e-5 + tf.reduce_sum(grad ** 2, 1))
        gp = tf.reduce_mean((norm - 1.) ** 2)
        print("gp_debug: grad.max {:E} grad.min {:E} norm.max {:E} norm.min {:E}), \n\
        ".format(tf.reduce_max(grad), tf.reduce_min(grad), tf.reduce_max(norm), tf.reduce_min(norm)))
        # del t
        return gp

    def my_0():
        return 0

    def my_1():
        return 1

    def my_ce(output, target):
        loss = 0
        for i in range(flags.batch_size):
            equal_front = tf.cond(
                tf.equal(tf.reduce_sum(tf.cast(tf.equal(target[i, 0:3], tf.constant([0., 0., 0.])), dtype=tf.int32)),
                         tf.constant(3, dtype=tf.int32)), my_1, my_0)
            equal_back = tf.cond(
                tf.equal(tf.reduce_sum(tf.cast(tf.equal(target[i, 3:5], tf.constant([0., 0.])), dtype=tf.int32)),
                         tf.constant(2, dtype=tf.int32)), my_1, my_0)
            if equal_front and (not equal_back):  # front = 0, 0, 0
                loss += tl.cost.binary_cross_entropy(output[i:i + 1, 3:5], target[i:i + 1, 3:5])
            elif (not equal_front) and equal_back:  # back = 0, 0
                loss += tl.cost.binary_cross_entropy(output[i:i + 1, 0:3], target[i:i + 1, 0:3])
            elif (not equal_front) and (not equal_back):
                loss += 0
            else:
                loss += tl.cost.binary_cross_entropy(output[i:i + 1, :], target[i:i + 1, :])
        loss = loss / flags.batch_size
        return loss

    def compute_cls_acc(output, target):  # (batch, 5)
        acc_hair = []
        acc_male = []

        for i in range(flags.batch_size):
            equal_front = tf.cond(
                tf.equal(tf.reduce_sum(tf.cast(tf.equal(target[i, 0:3], tf.constant([0., 0., 0.])), dtype=tf.int32)),
                         tf.constant(3, dtype=tf.int32)), my_1, my_0)
            equal_back = tf.cond(
                tf.equal(tf.reduce_sum(tf.cast(tf.equal(target[i, 3:5], tf.constant([0., 0.])), dtype=tf.int32)),
                         tf.constant(2, dtype=tf.int32)), my_1, my_0)
            if equal_front == 0:
                acc_hair.append(np.mean(np.equal(np.argmax(output.numpy()[i:i + 1, 0:3], axis=1),
                                                 np.argmax(target.numpy()[i:i + 1, 0:3], axis=1))))
            if equal_back == 0:
                acc_male.append(np.mean(np.equal(np.argmax(output.numpy()[i:i + 1, 3:5], axis=1),
                                                 np.argmax(target.numpy()[i:i + 1, 3:5], axis=1))))
        mean_acc_hair = np.mean(acc_hair)
        mean_acc_male = np.mean(acc_male)
        return mean_acc_hair, mean_acc_male

    for epoch in range(start_epoch, flags.n_epoch):
        start_time = time.time()
        n_iter = 1
        _acc_1 = 0
        _acc_2 = 0
        _acc_3 = 0
        _acc_m_hair = 0
        _acc_m_male = 0
        _acc_r_hair = 0
        _acc_r_male = 0
        print(epoch)
        # update the learning rate

        if epoch >= 10 and epoch % 10 == 0:
            new_lr = flags.lr_init - flags.lr_init * (15 - 10) / (20 - 10)
            #new_lr = lr_v / 2
            lr_v.assign(lr_v, new_lr)
            print("New learning rate %f" % new_lr)

        # sgd update params
        # for step, (data_pair_A, data_pair_B) in enumerate(zip(train_ds_A, train_ds_B)):
        for step, (data_pair_A, data_pair_B) in enumerate(zip(train_ds_A, train_ds_B)):
            if data_pair_A[0].shape[0] != flags.batch_size:  # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()

            Xa = data_pair_A[0]
            ta = data_pair_A[1]
            ta_m_r = data_pair_A[2]
            ta_r = data_pair_A[3]
            ta_gt = data_pair_A[4]
            ta_r_gt = data_pair_A[5]
            ta_e = tf.zeros_like(ta)

            Xb = data_pair_B[0]
            tb_r = data_pair_B[3]
            tb_gt = data_pair_B[4]
            tb_r_gt = data_pair_B[5]

            with tf.GradientTape(persistent=True) as tape:
                ## 1. data forward
                a_appearance = Ea(Xa)
                a_content = Ec(Xa)

                fake_Xa = G([a_appearance, a_content, ta_r])  # using relevant labels
                fake_a_content = Ec(fake_Xa)
                final_Xa = G([a_appearance, fake_a_content, ta])
                recon_Xa = G([a_appearance, a_content, ta])

                appearance_noise = np.random.normal(loc=0.0, scale=1.0, size=[flags.batch_size, flags.z_dim]).astype(
                    np.float32)
                d_Xa = G([appearance_noise, a_content, ta_e])
                d_a_appearance = Ea(d_Xa)

                b_appearance = Ea(Xb)
                e_X = G([b_appearance, a_content, tb_r])

                ## 2. loss
                L1_cc_loss = tl.cost.absolute_difference_error(final_Xa, Xa, is_mean=True)
                L1_recon_loss = tl.cost.absolute_difference_error(recon_Xa, Xa, is_mean=True)
                L1_app_loss = tl.cost.absolute_difference_error(d_a_appearance, appearance_noise, is_mean=True)

                Xa_logits, Xa_label = D(Xa)
                fake_Xa_logits, fake_Xa_label = D(fake_Xa)  # using random/relevant/any labels
                e_logits, e_label = D(e_X)

                Xa_acc_1 = np.mean(np.equal(Xa_logits.numpy() > 0.5, tf.ones_like(Xa_logits)))
                Xa_acc_2 = np.mean(np.equal(fake_Xa_logits.numpy() > 0.5, tf.zeros_like(fake_Xa_logits)))
                e_acc_3 = np.mean(np.equal(e_logits.numpy() > 0.5, tf.zeros_like(e_logits)))

                Xa_acc_hair, Xa_acc_male = compute_cls_acc(Xa_label, ta_gt)
                fake_Xa_acc_hair, fake_Xa_acc_male = compute_cls_acc(fake_Xa_label, ta_r_gt)

                d_logit_loss_real = - tf.reduce_mean(Xa_logits)
                d_logit_loss_fake = tf.reduce_mean(fake_Xa_logits)
                d_logits_loss_e_fake = tf.reduce_mean(e_logits)
                # d_label_loss = my_ce(Xa_label, ta)
                d_label_loss = tl.cost.binary_cross_entropy(Xa_label, ta_gt)

                # gradient penalty
                d_loss_gp = gradient_penalty(D, Xa, fake_Xa)
                d_loss_total = flags.lambda_d_logit_loss_real * d_logit_loss_real + \
                               flags.lambda_d_logit_loss_fake * d_logit_loss_fake + \
                               flags.lambda_d_label_loss * d_label_loss + \
                               flags.lambda_d_logit_loss_e_fake * d_logits_loss_e_fake + \
                               flags.lambda_gp * d_loss_gp

                ## Update G
                # g_label_loss = my_ce(fake_Xa_label, ta_r)
                g_label_loss = tl.cost.binary_cross_entropy(fake_Xa_label, ta_r_gt)
                g_logit_loss_fake = - tf.reduce_mean(fake_Xa_logits)
                g_logit_loss_e_fake = - tf.reduce_mean(e_logits)
                g_e_label_loss = tl.cost.binary_cross_entropy(e_label, tb_r_gt)

                g_loss_total = flags.lambda_g_logit_loss_fake * g_logit_loss_fake + \
                               flags.lambda_l1_cc_loss * L1_cc_loss + \
                               flags.lambda_l1_recon_loss * L1_recon_loss + \
                               flags.lambda_g_label_loss * g_label_loss + \
                               flags.lambda_l1_app_loss * L1_app_loss + \
                               flags.lambda_g_logit_loss_e_fake * g_logit_loss_e_fake + \
                               flags.lambda_g_e_label_loss * g_e_label_loss

            # compute gradients, update params
            grad = tape.gradient(d_loss_total, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            if (step + 1) % flags.critic_n == 0:
                grad = tape.gradient(g_loss_total,
                                     G.trainable_weights + Ec.trainable_weights + Ea.trainable_weights)  # DH: CA is a part of the generator
                g_optimizer.apply_gradients(
                    zip(grad, G.trainable_weights + Ec.trainable_weights + Ea.trainable_weights))
            # print("update time:{}".format(time.time() - st))

            del tape
            _acc_1 += Xa_acc_1
            _acc_2 += Xa_acc_2
            _acc_3 += e_acc_3
            mean_acc_1 = _acc_1 / n_iter
            mean_acc_2 = _acc_2 / n_iter
            mean_acc_3 = _acc_3 / n_iter

            _acc_m_hair += Xa_acc_hair
            _acc_m_male += Xa_acc_male
            _acc_r_hair += fake_Xa_acc_hair
            _acc_r_male += fake_Xa_acc_male
            mean_acc_m_hair = _acc_m_hair / n_iter
            mean_acc_m_male = _acc_m_male / n_iter
            mean_acc_r_hair = _acc_r_hair / n_iter
            mean_acc_r_male = _acc_r_male / n_iter
            n_iter += 1

            if np.mod(step, flags.print_every_step) == 0:
                print("Epoch:[{}/{}] [{}/{}] took:{:.4f}, \n\
                      g_loss: {:.4f} ({:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}), \n\
                      d_loss: {:.4f} ({:.4f} {:.4f} {:.4f} {:.4f} gp:{:.4f})".format(epoch, flags.n_epoch, step,
                                                                              n_step_epoch, time.time() - step_time,
                                                                              g_loss_total, L1_cc_loss, L1_recon_loss,
                                                                              L1_app_loss, g_logit_loss_fake,
                                                                              g_label_loss, g_e_label_loss, g_logit_loss_e_fake,
                                                                              d_loss_total, d_logit_loss_real,
                                                                              d_logit_loss_fake, d_label_loss, d_logits_loss_e_fake,
                                                                              d_loss_gp))
                print("acc:({:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}), \n\\".format(mean_acc_1, mean_acc_2, mean_acc_3,
                                                                                     mean_acc_m_hair, mean_acc_m_male,
                                                                                     mean_acc_r_hair, mean_acc_r_male))
                print("acc for this step:({:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}), \n\\".format(Xa_acc_1,
                                                                                                   Xa_acc_2,
                                                                                                   e_acc_3,
                                                                                                   Xa_acc_hair,
                                                                                                   Xa_acc_male,
                                                                                                   fake_Xa_acc_hair,
                                                                                                   fake_Xa_acc_male))
                # print("acc:({:.4f} {:.4f} {:.4f}), \n\\".format(mean_acc_1, mean_acc_2, mean_acc_3))

            if np.mod(step, flags.save_every_step) == 0:
                test_image_1(epoch, step)
                test_image_2(epoch, step)

            if np.mod(step, flags.save_every_step * 10) == 0:
                for i in range(10):
                    test_image_3(epoch, step, i)
                for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    test_image_4(epoch, step, alpha)
            if np.mod(step, flags.save_weight_every_step) == 0:
                G.save_weights('{}/{}/G_{}.h5'.format(flags.model_dir, flags.prefix, epoch))
                D.save_weights('{}/{}/D_{}.h5'.format(flags.model_dir, flags.prefix, epoch))
                Ea.save_weights('{}/{}/Ea_{}.h5'.format(flags.model_dir, flags.prefix, epoch))
                Ec.save_weights('{}/{}/Ec_{}.h5'.format(flags.model_dir, flags.prefix, epoch))
            # CA.save_weights('{}/{}/CA_{}.h5'.format(flags.model_dir, flags.prefix, str(epoch)))  # DH: Save C

            # test_image_2(epoch)
            # for i in range(20):
            #  test_image_3(epoch, i)


if __name__ == '__main__':
    train()
