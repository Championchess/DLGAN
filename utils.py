# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorlayer as tl


def KL_loss(mu, log_sigma):
    loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(2 * log_sigma) - 1 - 2 * log_sigma)
    loss = tf.reduce_mean(loss)
    loss = tf.reduce_mean(loss)
    return loss


# save picked labels to a file
def save_labels(filename, labels_list, label_names, item_names = ['downblack', 'downwhite', 'downgray', 
                     'downblue', 'upblack', 'upwhite',
                    'upgray', 'upblue']):
    f = open(filename, 'w')
    s = ' '
    for i in item_names:
        s += (i + ' ')
    f.write(s + '\n')
    for i in range(len(labels_list)):
        f.write(label_names[i] + '\n')
        for j in range(25):
            s = ''
            s += (str(j) + ': ')
            for k in range(8):
                if int(labels_list[i][j][k]) == 1:
                    s += (item_names[k] + ' ')
            f.write(s + '\n')
    print('Successfully saving labels.')


def image_aug_fn(x):
    if flags.dataset == 'celebA':
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
    
    elif flags.dataset == 'DukeMTMC':
        return tf.image.random_flip_left_right(x)


def image_aug_fn_for_test(x):
    if flags.dataset == 'celebA':
        x = tl.prepro.imresize(x, [flags.im_sz, flags.im_sz])
        x = x / 127.5 - 1.
        x = x.astype(np.float32)
        return x
    
    elif flags.dataset == 'DukeMTMC':
        return x
