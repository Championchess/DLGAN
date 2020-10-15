# -*- coding: utf-8 -*-
"""
Created on Sat 20/7/2019

@original: zyh
"""
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import (BatchNorm2d, Conv2d, Dense, Flatten, Input, DeConv2d, Lambda, \
                                LocalResponseNorm, MaxPool2d, Elementwise, InstanceNorm2d, \
                                Concat, ExpandDims, Tile, GlobalMeanPool2d, UpSampling2d, \
                                MeanPool2d, GaussianNoise, LayerNorm)
from tensorlayer.models import Model
from data import flags

'''
latest version: author Zbc 2019 08 15 23:47
'''


# w_init = tf.random_normal_initializer(stddev=0.02)
# g_init = tf.random_normal_initializer(1., 0.02)

def spectral_norm(w, u,
                  iteration=1):  # https://github.com/taki0112/Spectral_Normalization-Tensorflow/blob/master/spectral_norm.py
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    # u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)
    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


class SpectralNormConv2d(Conv2d):
    """
    The :class:`SpectralNormConv2d` class is a Conv2d layer for with Spectral Normalization.
    ` Spectral Normalization for Generative Adversarial Networks (ICLR 2018) <https://openreview.net/forum?id=B1QRgziT-&noteId=BkxnM1TrM>`__

    Parameters
    ----------
    n_filter : int
    The number of filters.
    filter_size : tuple of int
    The filter size (height, width).
    strides : tuple of int
    The sliding window strides of corresponding input dimensions.
    It must be in the same order as the ``shape`` parameter.
    dilation_rate : tuple of int
    Specifying the dilation rate to use for dilated convolution.
    act : activation function
    The activation function of this layer.
    padding : str
    The padding algorithm type: "SAME" or "VALID".
    data_format : str
    "channels_last" (NHWC, default) or "channels_first" (NCHW).
    W_init : initializer
    The initializer for the the weight matrix.
    b_init : initializer or None
    The initializer for the the bias vector. If None, skip biases.
    in_channels : int
    The number of in channels.
    name : None or str
    A unique layer name.
    """

    def __init__(
            self,
            n_filter=32,
            filter_size=(3, 3),
            strides=(1, 1),
            act=None,
            padding='SAME',
            data_format='channels_last',
            dilation_rate=(1, 1),
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            in_channels=None,
            name=None
    ):
        super(SpectralNormConv2d, self).__init__(n_filter=n_filter, filter_size=filter_size,
                                                 strides=strides, act=act, padding=padding, data_format=data_format,
                                                 dilation_rate=dilation_rate, W_init=W_init, b_init=b_init,
                                                 in_channels=in_channels,
                                                 name=name)
        # logging.info(
        #     "    It is a SpectralNormConv2d %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s" % (
        #         self.name, n_filter, str(filter_size), str(strides), padding,
        #         self.act.__name__ if self.act is not None else 'No Activation'
        #     )
        # )
        if self.in_channels:
            self.build(None)
            self._built = True

    def build(self, inputs_shape):  # # override
        self.u = self._get_weights("u", shape=[1, self.n_filter], init=tf.random_normal_initializer(),
                                   trainable=False)  # tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
        # self.s =  self._get_weights("sigma", shape=[1, ], init=tf.random_normal_initializer(), trainable=False)
        super(SpectralNormConv2d, self).build(inputs_shape)

    def forward(self, inputs):  # override
        self.W_norm = spectral_norm(self.W, self.u)
        # self.W_norm = spectral_norm(self.W, self.u, self.s)
        # return super(SpectralNormConv2d, self).forward(inputs)
        outputs = tf.nn.conv2d(
            input=inputs,
            filters=self.W_norm,
            strides=self._strides,
            padding=self.padding,
            data_format=self.data_format,  # 'NHWC',
            dilations=self._dilation_rate,  # [1, 1, 1, 1],
            name=self.name,
        )
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, data_format=self.data_format, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs


# t_dim = 128                 # text feature dimension
# rnn_hidden_size = t_dim
# vocab_size = 8000
# word_embedding_size = 256
# def txt_embed(x):
#     LSTMCell = tf.contrib.rnn.BasicLSTMCell
#     net = tl.layers.EmbeddingInputlayer(
#                  vocabulary_size = vocab_size,
#                  embedding_size = word_embedding_size,
#                  E_init = w_init,
#                  name = 'wordembed')
#     net = tl.layers.DynamicRNNLayer(net,
#                  cell_fn = LSTMCell,
#                  cell_init_args = {'state_is_tuple' : True, 'reuse': reuse},
#                  n_hidden = rnn_hidden_size,
#                  dropout = None,#(keep_prob if is_train else None),
#                  initializer = w_init,
#                  sequence_length = tl.layers.retrieve_seq_length_op2(x),
#                  # sequence_length = tl.layers.retrieve_seq_length_op(x),
#                  return_last = True,
#                  name = 'dynamicrnn')
#     return net

# data = [[1, 2, 0, 0, 0], [1, 2, 3, 0, 0], [1, 2, 6, 1, 1]]
# data = tf.convert_to_tensor(data, dtype=tf.int32)
# print(tl.layers.retrieve_seq_length_op3(data))
#
# class DynamicRNNExample(tl.models.Model):
#     def __init__(self):
#         super(DynamicRNNExample, self).__init__()
#         self.embedding = tl.layers.Embedding(600, 128)
#         self.rnnlayer = tl.layers.RNN(
#                    cell=tf.keras.layers.LSTMCell(units=128), in_channels=128,
#                    return_last_output=True, return_last_state=True)
#     def forward(self, x):
#         z, s = self.rnnlayer(self.embedding(x), sequence_length=tl.layers.retrieve_seq_length_op3(x))
#         return z, s
# model = DynamicRNNExample()
# model.eval()
# output, state = model(data)
# print(output)
# exit()

class get_RNN(tl.models.Model):
    def __init__(self, vocab_size=flags.vocab_size, word_embedding_size=flags.word_embedding_size,
                 rnn_hidden_size=flags.t_dim):
        super(get_RNN, self).__init__()
        self.embedding = tl.layers.Embedding(vocab_size, word_embedding_size)
        self.rnnlayer = tl.layers.RNN(
            cell=tf.keras.layers.LSTMCell(units=rnn_hidden_size), in_channels=word_embedding_size,
            return_last_output=True, return_last_state=True)

    def forward(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.int32)  # if you don't use dataset api
        # print(type(x))
        # print(tl.layers.retrieve_seq_length_op3(x))
        out, state = self.rnnlayer(self.embedding(x), sequence_length=tl.layers.retrieve_seq_length_op3(x))
        return out  # , state


# for pretrain RNN, input image output vector
def get_CNN():
    df_dim = 64
    act = tf.nn.relu

    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)

    # ni = Input((None, 64, 64, 3))
    ni = Input((None, 256, 256, 3))  # add for image_size
    n = Conv2d(df_dim, (4, 4), (2, 2), act=act,
               padding='SAME', W_init=w_init)(ni)

    n = Conv2d(df_dim * 2, (4, 4), (2, 2), W_init=w_init, b_init=None)(n)  # add for 256
    n = BatchNorm2d(decay=0.99, act=act, gamma_init=g_init)(n)

    n = Conv2d(df_dim * 2, (4, 4), (2, 2), W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(decay=0.99, act=act, gamma_init=g_init)(n)

    n = Conv2d(df_dim * 4, (4, 4), (2, 2), W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(decay=0.99, act=act, gamma_init=g_init)(n)

    n = Conv2d(df_dim * 8, (4, 4), (2, 2), W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(decay=0.99, act=act, gamma_init=g_init)(n)

    n = Conv2d(df_dim * 16, (4, 4), (2, 2), W_init=w_init, b_init=None)(n)  # add for 256
    n = BatchNorm2d(decay=0.99, act=act, gamma_init=g_init)(n)

    n = Flatten()(n)
    n = Dense(n_units=flags.t_dim, W_init=w_init)(n)
    return tl.models.Model(inputs=ni, outputs=n)


## content encoder: input X, output c   from Yuanhao Qi's Work
## origin
# def get_Ec(x_shape=(None, flags.im_sz, flags.im_sz, 3), c_shape = \
#           (None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]), \
#           name=None):
#    # ref: Multimodal Unsupervised Image-to-Image Translation
#    lrelu = lambda x: tl.act.lrelu(x, 0.01)
#    w_init = tf.random_normal_initializer(stddev=0.02)
#    channel = 64
#    ni = Input(x_shape)
#    n = Conv2d(64, (7,7), (1,1), act=tf.nn.relu, W_init=w_init)(ni)
#    n = Conv2d(128, (4,4), (2,2), act=tf.nn.relu, W_init=w_init)(n)
#    n = Conv2d(256, (4,4), (2,2), act=tf.nn.relu, W_init=w_init)(n)
#
#    nn = Conv2d(256, (3,3), (1,1), act=tf.nn.relu, W_init=w_init)(n)
#    nn = Conv2d(256, (3,3), (1,1), act=None, W_init=w_init)(nn)
#    n = Elementwise(tf.add)([n, nn])
#
#    nn = Conv2d(256, (3,3), (1,1), act=tf.nn.relu, W_init=w_init)(n)
#    nn = Conv2d(256, (3,3), (1,1), act=None, W_init=w_init)(nn)
#    n = Elementwise(tf.add)([n, nn])
#
#    nn = Conv2d(256, (3,3), (1,1), act=tf.nn.relu, W_init=w_init)(n)
#    nn = Conv2d(256, (3,3), (1,1), act=None, W_init=w_init)(nn)
#    n = Elementwise(tf.add)([n, nn])
#
#    nn = Conv2d(256, (3,3), (1,1), act=tf.nn.relu, W_init=w_init)(n)
#    nn = Conv2d(256, (3,3), (1,1), act=None, W_init=w_init)(nn)
#    n = Elementwise(tf.add)([n, nn])
#
#    M = Model(inputs=ni, outputs=n, name=name)
#    return M

## content encoder: input X, output c
## ref: TAGAN encoder to 16*16*512
# def get_Ec(x_shape=(None, flags.im_sz, flags.im_sz, 3), c_shape = \
#           (None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]), \
#           name=None):
#    # ref: Multimodal Unsupervised Image-to-Image Translation
#    lrelu = lambda x: tl.act.lrelu(x, 0.01)
#    w_init = tf.random_normal_initializer(stddev=0.02)
#    g_init = tf.random_normal_initializer(1., 0.02)
#    channel = 64
#    ni = Input(x_shape)
#
#    nn = Conv2d(channel, (3,3), (1,1), act=tf.nn.relu, W_init=w_init)(ni)
#
#    nn = Conv2d(channel*2, (3,3), (2,2), act=None, W_init=w_init)(nn)
#    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
#    nn = Conv2d(channel*4, (3,3), (2,2), act=None, W_init=w_init)(nn)
#    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
#    nn = Conv2d(channel*8, (3,3), (2,2), act=None, W_init=w_init)(nn)
#    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
#
#    M = Model(inputs=ni, outputs=nn, name=name)
#    return M

# content encoder: input X, output c
# ref: DRIT Ec 128*128->32*32*256
def get_Ec(x_shape=(None, flags.im_h, flags.im_w, 3), c_shape= \
        (None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]), \
           name=None):
    lrelu = lambda x: tl.act.lrelu(x, 0.01)
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)
    channel = 64  # 128
    ni = Input(x_shape)
    # (1, 128, 128, 3)
    n = Conv2d(channel, (7, 7), (1, 1), act=lrelu, W_init=w_init)(ni)

    # channel = channel * 2 # temp
    # n = Conv2d(channel, (3, 3), (1, 1), W_init=w_init, b_init=None)(n) # temp
    # n = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(n) # temp

    for i in range(2):
        n = Conv2d(channel * 2, (3, 3), (2, 2), W_init=w_init, b_init=None)(n)
        # n = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(n)
        n = BatchNorm2d(decay=0.99, act=tf.nn.relu, gamma_init=g_init)(n)
        channel = channel * 2

    for i in range(0, 3):
        # res block
        nn = Conv2d(channel, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
        # nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
        nn = BatchNorm2d(decay=0.99, act=tf.nn.relu, gamma_init=g_init)(nn)
        nn = Conv2d(channel, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
        # nn = InstanceNorm2d(act=None, gamma_init=g_init)(nn)
        nn = BatchNorm2d(decay=0.99, act=None, gamma_init=g_init)(nn)
        n = Elementwise(tf.add, act=tf.nn.relu)([n, nn])
    #print(ni.shape, n.shape)
    # n = GaussianNoise(is_always=False)(n)
    M = Model(inputs=ni, outputs=n, name=name)
    return M


## content encoder: input X, output c
## ref: DRIT Ec 64*64->16*16*512
# def get_Ec(x_shape=(None, flags.im_sz, flags.im_sz, 3), c_shape= \
#        (None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]), \
#           name=None):
#    lrelu = lambda x: tl.act.lrelu(x, 0.01)
#    w_init = tf.random_normal_initializer(stddev=0.02)
#    g_init = tf.random_normal_initializer(1., 0.02)
#    channel = 64 #128
#    ni = Input(x_shape)
#    # (1, 64, 64, 3)
#    n = Conv2d(channel, (3, 3), (1, 1), act=lrelu, W_init=w_init)(ni)
#
#    channel = channel * 2 # temp
#    n = Conv2d(channel, (3, 3), (1, 1), W_init=w_init, b_init=None)(n) # temp
#    n = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(n) # temp
#
#    # (64, 64, 64, 3)
#    for i in range(2):
#        n = Conv2d(channel * 2, (3, 3), (2, 2), W_init=w_init, b_init=None)(n)
#        # n = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(n)
#        n = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(n)
#        channel = channel * 2
#
#    for i in range(0, 3):
#        # res block
#        nn = Conv2d(channel, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
#        # nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
#        nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
#        nn = Conv2d(channel, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
#        # nn = InstanceNorm2d(act=None, gamma_init=g_init)(nn)
#        nn = BatchNorm2d(decay=0.9, act=None, gamma_init=g_init)(nn)
#        n = Elementwise(tf.add, act=tf.nn.relu)([n, nn])
#
#    # n = GaussianNoise(is_always=False)(n)
#    M = Model(inputs=ni, outputs=n, name=name)
#    return M


# left concat
## appearance encoder: input X, output z (including z, mean, log_sigma)
# def get_Ea(x_shape=(None, 256, 256, 3), z_shape=(None, flags.z_dim), \
#           name=None):
#    # ref: Multimodal Unsupervised Image-to-Image Translation
#
#    w_init = tf.random_normal_initializer(stddev=0.02)
#
#    ni = Input(x_shape)
#
#    n = Conv2d(64, (7,7), (1,1), act=tf.nn.relu, W_init=w_init)(ni)
#    n = Conv2d(128, (4,4), (2,2), act=tf.nn.relu, W_init=w_init)(n)
#    n = Conv2d(256, (4,4), (2,2), act=tf.nn.relu, W_init=w_init)(n)
#    n = Conv2d(256, (4,4), (2,2), act=tf.nn.relu, W_init=w_init)(n)
#    n = GlobalMeanPool2d()(n)
#    n = Flatten()(n)
#    mean = Dense(flags.z_dim)(n)
#    log_sigma = Dense(flags.z_dim)(n)
#    def sample(mean, log_sigma):
#        epsilon = tf.random.truncated_normal(mean.shape)
#        stddev = tf.exp(log_sigma)
#        out = mean + stddev * epsilon
#        return out
#    no = Lambda(sample)([mean, log_sigma])
#    M = Model(inputs=ni, outputs=[no, mean, log_sigma], name=name)
#    return M

# DRIT version
# def get_Ea(x_shape=(None, flags.im_sz, flags.im_sz, 3), z_shape=(None, flags.z_dim), name=None):
#     # input: (batch_size, im_sz, im_sz, 3)
#     # ref: DRIT  Ea_concat
#
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     g_init = tf.random_normal_initializer(1., 0.02)
#     ndf = 64
#     lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
#
#     ni = Input(x_shape)
#     n = Conv2d(ndf, (3, 3), (1, 1), padding='VALID', W_init=w_init, act=None)(ni)
#     # ResBlock implementation details
#     # add(shortcut+conv) first add, then act
#     # last layer of shortcut/conv doesn't have act(None)
#     # shortcut doesn't have bn
#
#     ## Basic Blocks * 4
#     for i in range(1, 4):
#         ## Basic Block
#         # conv part
#         n = Conv2d(ndf * i, (3, 3), (1, 1), padding='VALID', W_init=w_init, b_init=None, act=None)(n)  # conv3x3
#         n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#         n = Conv2d(ndf * (i + 1), (3, 3), (1, 1), padding='VALID', W_init=w_init, b_init=None, act=None)(n)  # conv3x3 in convMeanpool
#         n = BatchNorm2d(decay=0.9, act=None, gamma_init=g_init)(n)
#         n = MeanPool2d((2, 2), (2, 2))(n)  # meanPool2d in convMeanpool
#         # shortcut part
#         ns = MeanPool2d((2, 2), (2, 2))(n)
#         ns = Conv2d(ndf * (i + 1), (3, 3), (1, 1), padding='VALID', W_init=w_init, act=None)(ns)
#         n = Elementwise(tf.add, act=lrelu)([n, ns])
#
#     n = GlobalMeanPool2d()(n)
#     no = Dense(n_units=flags.z_dim, W_init=w_init, name="mean_linear")(n)
#     M = Model(inputs=ni, outputs=no, name=name)
#     return M

# def get_Ea(x_shape=(None, flags.im_sz, flags.im_sz, 3), z_shape=(None, flags.z_dim), name=None):
#    # input: (batch_size, im_sz, im_sz, 3)
#    # ref: DRIT Ea
#
#    w_init = tf.random_normal_initializer(stddev=0.02)
#    g_init = tf.random_normal_initializer(1., 0.02)
#    ndf = 64
#    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
#
#    ni = Input(x_shape)
#    nn = Conv2d(ndf, (3, 3), (1, 1), padding='SAME', W_init=w_init, act=tf.nn.relu)(ni)
#
##    nn = Conv2d(ndf*2, (4,4), (2,2), padding='SAME', W_init=w_init, act=tf.nn.relu)(nn)
##    nn = Conv2d(ndf*4, (4,4), (2,2), padding='SAME', W_init=w_init, act=tf.nn.relu)(nn)
##    nn = Conv2d(ndf*4, (4,4), (2,2), padding='SAME', W_init=w_init, act=tf.nn.relu)(nn)
##    nn = Conv2d(ndf*4, (4,4), (2,2), padding='SAME', W_init=w_init, act=tf.nn.relu)(nn)
#
#    nn = Conv2d(ndf*2, (4,4), (2,2), padding='SAME', W_init=w_init, b_init=None)(nn)
#    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
#    nn = Conv2d(ndf*4, (4,4), (2,2), padding='SAME', W_init=w_init, b_init=None)(nn)
#    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
#    nn = Conv2d(ndf*4, (4,4), (2,2), padding='SAME', W_init=w_init, b_init=None)(nn)
#    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
#    nn = Conv2d(ndf*4, (4,4), (2,2), padding='SAME', W_init=w_init, b_init=None)(nn)
#    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
#    n = GlobalMeanPool2d()(nn)
#
#    n = Dense(flags.z_dim, W_init=w_init, act=None)(n)
#
#    M = Model(inputs=ni, outputs=n, name=name)
#    return M

# 128*128-> 128
def get_Ea(x_shape=(None, flags.im_h, flags.im_w, 3), z_shape=(None, flags.z_dim), name=None):
    # input: (batch_size, im_sz, im_sz, 3)
    # ref: DRIT  Ea_concat

    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)
    ndf = 64
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    ni = Input(x_shape)
    nn = Conv2d(ndf, (7, 7), (1, 1), padding='SAME', W_init=w_init, act=lrelu)(ni)

    ## Basic Blocks * 3
    for i in range(1, 4):
        ## Basic Block
        # conv part
        # n = Lambda(lrelu)(nn)  # leaky relu (0.2)
        n = Conv2d(ndf * i, (3, 3), (1, 1), padding='VALID', W_init=w_init, b_init=None, act=None)(nn)  # conv3x3
        n = BatchNorm2d(decay=0.99, act=lrelu, gamma_init=g_init)(n)
        # n = InstanceNorm2d(act=lrelu, gamma_init=g_init)(n)
        n = Conv2d(ndf * (i + 1), (3, 3), (1, 1), padding='VALID', W_init=w_init, b_init=None, act=None)(n)  # conv3x3 in convMeanpool
        n = BatchNorm2d(decay=0.99, act=None, gamma_init=g_init)(n)
        # n = InstanceNorm2d(act=None, gamma_init=g_init)(n)
        n = MeanPool2d((2, 2), (2, 2))(n)  # meanPool2d in convMeanpool
        # shortcut part
        ns = MeanPool2d((2, 2), (2, 2))(nn)
        ns = Conv2d(ndf * (i + 1), (3, 3), (1, 1), padding='VALID', b_init=None, W_init=w_init, act=None)(ns)
        nn = Elementwise(tf.add, act=lrelu)([n, ns])

    n = GlobalMeanPool2d()(nn)

    n = Dense(flags.z_dim, W_init=w_init, act=None)(n)
    #print(ni.shape, n.shape)
    M = Model(inputs=ni, outputs=n, name=name)
    return M


# DRIT generator
# def get_G(a_shape=(None, flags.z_dim), \
#           c_shape=(None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]), \
#           t_shape=(None, flags.t_dim), \
#           # nt_params=None,
#           name=None):
#     lrelu = lambda x: tl.act.lrelu(x, 0.2)
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     g_init = tf.random_normal_initializer(1., 0.02)
#     ndf = 512
#     na = Input(a_shape)
#     nc = Input(c_shape)  # (1, 16, 16, 512)
#     nt = Input(t_shape)
#     #nt = Dense(n_units=flags.t_dense_dim, act=lrelu, W_init=w_init)(nt)  # add dense layer for t_input
#     z = Concat(-1)([na, nt])  # (1, 136) = (1, 128) + (1, 8)
#     nz = ExpandDims(1)(z)
#     nz = ExpandDims(1)(nz)  # (1, 1, 1, 136)
#
#     nz = Tile([1, c_shape[1], c_shape[2], 1])(nz)  # (1, 16, 16, 136)
#
#     n = Concat(-1)([nc, nz])
#
#     n = Conv2d(ndf, (3,3), (1,1), act=None, W_init=w_init, b_init=None)(n)
#     n = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(n)
#
#     # res block *4
#     for i in range(4):
#         nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
#         # nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
#         nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
#         nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
#         # nn = InstanceNorm2d(act=None, gamma_init=g_init)(nn)
#         nn = BatchNorm2d(decay=0.9, act=None, gamma_init=g_init)(nn)
#         n = Elementwise(tf.add)([n, nn])
#
#     for i in range(2):
#         n = DeConv2d(ndf // 2, (4, 4), (2, 2), act=None, W_init=w_init, b_init=None)(n)
#         # n = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(n)
#         n = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(n)
#
#         ndf = ndf // 2
#
#     n = Conv2d(3, (3, 3), (1, 1), act=tf.nn.tanh, W_init=w_init)(n)
#     M = Model(inputs=[na, nc, nt], outputs=n, name=name)
#     return M


# generator: input a, c, txt, output X', encode 16*16*512
# ref: DRIT+stargan DH
# def get_G(a_shape=(None, flags.z_dim), \
#           c_shape=(None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]), \
#           t_shape=(None, flags.t_dim), \
#           # nt_params=None,
#           name=None):
#     lrelu = lambda x: tl.act.lrelu(x, 0.2)
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     g_init = tf.random_normal_initializer(1., 0.02)
#     ndf = 512
#     na = Input(a_shape)
#     nc = Input(c_shape)  # (1, 16, 16, 512)
#     nt = Input(t_shape)
#     # nt = Dense(n_units=flags.t_dense_dim, act=lrelu, W_init=w_init)(nt)  # add dense layer for t_input
#     nat = Concat(-1)([na, nt])  # (1, 133) = (1, 128) + (1, 5)
#     nat = ExpandDims(1)(nat)
#     nat = ExpandDims(1)(nat)  # (1, 1, 1, 133)
#
#     nat = Tile([1, flags.c_shape[0], flags.c_shape[1], 1])(nat)  # (1, 16, 16, 136)
#
#     n = Concat(-1)([nat, nc])
#
#     nd_tmp = flags.z_dim + flags.t_dim
#     ndf = ndf + nd_tmp
#
#     # n = Concat(-1)([n, nat])
#
#     # res block *6
#     for i in range(6):
#         nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
#         # nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
#         nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
#         nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
#         # nn = InstanceNorm2d(act=None, gamma_init=g_init)(nn)
#         nn = BatchNorm2d(decay=0.9, act=None, gamma_init=g_init)(nn)
#         n = Elementwise(tf.add, act=lrelu)([n, nn])
#
#     for i in range(3):
#         n = DeConv2d(ndf // 2, (3, 3), (2, 2), act=None, W_init=w_init, b_init=None)(n)
#         # n = UpSampling2d((2, 2))(n)
#         # n = Conv2d(ndf//2, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
#         # n = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(n)
#         n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#
#         ndf = ndf // 2
#
#     # n = Concat(-1)([n, nat])
#     n = Conv2d(3, (1, 1), (1, 1), act=tf.nn.tanh, W_init=w_init)(n)
#     M = Model(inputs=[na, nc, nt], outputs=n, name=name)
#     return M


# generator: input a, c, txt, output X', encode 32*32*256
# ref: DRIT+stargan DH (split the t vector)
def get_G(a_shape=(None, flags.z_dim),
          c_shape=(None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]),
          t_shape=(None, flags.t_dim),
          # nt_params=None,
          name=None):
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)
    ndf = 256

    na = Input(a_shape)
    nc = Input(c_shape)  # (1, 32, 32, 512)
    nt = Input(t_shape)
    nat = Concat(-1)([na, nt])  # (1, 133) = (1, 128) + (1, 5)
    nat = ExpandDims(1)(nat)  # (1, 1, 133)
    nat = ExpandDims(1)(nat)  # (1, 1, 1, 133)
    nat = Tile([1, flags.c_shape[0], flags.c_shape[1], 1])(nat)  # (1, 32, 32, 133)
    n = Concat(-1)([nat, nc])  # (1, 32, 32, 645)
    nd_tmp = flags.z_dim + flags.t_dim  # 133
    ndf = ndf + nd_tmp  # 512+133=645

    # res block *6
    for i in range(6):
        nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
        # nn = InstanceNorm2d(act=lrelu, gamma_init=g_init)(nn)
        nn = BatchNorm2d(decay=0.99, act=lrelu, gamma_init=g_init)(nn)
        nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
        # nn = InstanceNorm2d(act=None, gamma_init=g_init)(nn)
        nn = BatchNorm2d(decay=0.99, act=None, gamma_init=g_init)(nn)
        n = Elementwise(tf.add, act=lrelu)([n, nn])

    ngt = ExpandDims(1)(nt)
    ngt = ExpandDims(1)(ngt)
    ngt = Tile([1, flags.c_shape[0], flags.c_shape[1], 1])(ngt)  # (1, 16, 16, 5)
    n = Concat(-1)([n, ngt])  # (1, 32, 32, 650)
    ndf += flags.t_dim
    
    for i in range(2):
        n = DeConv2d(ndf // 2, (3, 3), (2, 2), act=None, W_init=w_init, b_init=None)(n)
        #print("step"+str(i)+":"+str(n.shape))
        # n = UpSampling2d((2, 2))(n)
        # n = Conv2d(ndf//2, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
        # n = InstanceNorm2d(act=lrelu, gamma_init=g_init)(n)
        n = BatchNorm2d(decay=0.99, act=lrelu, gamma_init=g_init)(n)

        ndf = ndf // 2

    #print(n.shape)
    # n = Concat(-1)([n, nat])
    n = Conv2d(3, (1, 1), (1, 1), act=tf.nn.tanh, W_init=w_init)(n)
    #print(na.shape, nc.shape, nt.shape, n.shape)
    M = Model(inputs=[na, nc, nt], outputs=n, name=name)
    return M


## generator: input a, c, txt, output X', encode 16*16*512
## ref: DRIT generator-concat
# def get_G(a_shape=(None, flags.z_dim), \
#          c_shape=(None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]), \
#          t_shape=(None, flags.t_dim), \
#          # nt_params=None,
#          name=None):
#    lrelu = lambda x: tl.act.lrelu(x, 0.2)
#    w_init = tf.random_normal_initializer(stddev=0.02)
#    g_init = tf.random_normal_initializer(1., 0.02)
#    ndf = 512
#    na = Input(a_shape)
#    nc = Input(c_shape)  # (1, 16, 16, 512)
#    nt = Input(t_shape)
#    # nt = Dense(n_units=flags.t_dense_dim, act=lrelu, W_init=w_init)(nt)  # add dense layer for t_input
#    z = Concat(-1)([na, nt])  # (1, 136) = (1, 128) + (1, 8)
#    nz = ExpandDims(1)(z)
#    nz = ExpandDims(1)(nz)  # (1, 1, 1, 136)
#
#    nz = Tile([1, c_shape[1], c_shape[2], 1])(nz)  # (1, 16, 16, 136)
#
#    # print(nz.shape) # (1, 16, 16, 136)
#    # res block
#    nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nc)
#    # nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
#    nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
#
#    nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
#    # nn = InstanceNorm2d(act=None, gamma_init=g_init)(nn)
#    nn = BatchNorm2d(decay=0.9, act=None, gamma_init=g_init)(nn)
#
#    n = Elementwise(tf.add)([nc, nn])
#
#    nd_tmp = flags.z_dim + flags.t_dim
#    ndf = ndf + nd_tmp
#
#    n = Concat(-1)([n, nz])
#
#    # res block *3
#    for i in range(3):
#        nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
#        # nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
#        nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
#        nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
#        # nn = InstanceNorm2d(act=None, gamma_init=g_init)(nn)
#        nn = BatchNorm2d(decay=0.9, act=None, gamma_init=g_init)(nn)
#        n = Elementwise(tf.add)([n, nn])
#
#    for i in range(2):
#        ndf = ndf + nd_tmp
#        n = Concat(-1)([n, nz])
#        nz = Tile([1, 2, 2, 1])(nz)
#        n = DeConv2d(ndf // 2, (3, 3), (2, 2), act=None, W_init=w_init, b_init=None)(n)
#        # n = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(n)
#        n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#
#        ndf = ndf // 2
#
#    n = Concat(-1)([n, nz])
#    n = Conv2d(3, (1, 1), (1, 1), act=tf.nn.tanh, W_init=w_init)(n)
#    M = Model(inputs=[na, nc, nt], outputs=n, name=name)
#    return M


##generator: input a, c, txt, output X', encode 64*64*256
##ref: DRIT generator-concat
# def get_G(a_shape=(None, flags.z_dim), \
#          c_shape=(None, 64, 64, 256), \
#          t_shape=(None, flags.t_dim), \
#          #nt_params=None,
#          name=None):
#    w_init = tf.random_normal_initializer(stddev=0.02)
#    g_init = tf.random_normal_initializer(1., 0.02)
#    ndf = 256
#    na = Input(a_shape)
#    nc = Input(c_shape)
#    nt = Input(t_shape)
#    z = Concat(-1)([na, nt])
#    nz = ExpandDims(1)(z)
#    nz = ExpandDims(1)(nz)
#    nz = Tile([1, c_shape[1], c_shape[2], 1])(nz)
#
#    #res block
#    nn = Conv2d(ndf, (3,3), (1,1), act=None, W_init=w_init, b_init=None)(nc)
#    nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
#    nn = Conv2d(ndf, (3,3), (1,1), act=None, W_init=w_init, b_init=None)(nn)
#    nn = InstanceNorm2d(act=None, gamma_init=g_init)(nn)
#    n = Elementwise(tf.add)([nc, nn])
#
#    nd_tmp = flags.z_dim + flags.t_dim
#    ndf = ndf + nd_tmp
#    n= Concat(-1)([n, nz])
#
#    #res block *3
#    for i in range(1, 4):
#        nn = Conv2d(ndf, (3,3), (1,1), act=None, W_init=w_init, b_init=None)(n)
#        nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
#        nn = Conv2d(ndf, (3,3), (1,1), act=None, W_init=w_init, b_init=None)(nn)
#        nn = InstanceNorm2d(act=None, gamma_init=g_init)(nn)
#        n = Elementwise(tf.add)([n, nn])
#
#    for i in range(2):
#        ndf = ndf + nd_tmp
#        n = Concat(-1)([n, nz])
#        nz = Tile([1, 2, 2, 1])(nz)
#
#        n = DeConv2d(ndf//2, (3,3), (2,2), act=tf.nn.relu, W_init=w_init, b_init=None)(n)
#        n = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(n)
#
#        ndf = ndf // 2
#
#    n = Concat(-1)([n, nz])
#    n = DeConv2d(3, (1,1), (1,1), act=tf.nn.tanh, W_init=w_init)(n)
#
#    M = Model(inputs=[na, nc, nt], outputs=n, name=name)
#    return M


class get_CA(Model):  # conditional augmentation for text embedding (can be seem as an apart of G)
    def __init__(self):
        super(get_CA, self).__init__()
        lrelu = lambda x: tl.act.lrelu(x, 0.2)
        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)
        # no activation
        self.dense1 = Dense(n_units=flags.t_dim, W_init=w_init, in_channels=flags.t_dim)
        self.dense2 = Dense(n_units=flags.t_dim, W_init=w_init, in_channels=flags.t_dim)

    def forward(self, t_embed, disable_aug=False):
        # tt_shape = []
        # for tt in t_shape:
        #    tt.astype(float32)
        #    tt_shape.append(tt)
        # t_shape = tt_shape

        mean = self.dense1(t_embed)
        log_sigma = self.dense2(t_embed)

        if disable_aug:
            t_embed_vae = mean
        else:
            epsilon = tf.random.truncated_normal(mean.shape)
            stddev = tf.exp(log_sigma)
            t_embed_vae = mean + stddev * epsilon

        # M = self.G([a_embed, c_embed, t_embed_vae])#a_shape, c_shape, c, self.dense.all_weights)
        return t_embed_vae, mean, log_sigma


# ref: stargan D image 128*128
def get_D(x_shape=(None, flags.im_h, flags.im_w, 3)):
    df_dim = 64  # 64 for flower, 196 for MSCOCO
    s = flags.im_sz  # output image size [128]
    ks = int(flags.im_sz / 64)
    lrelu = lambda x: tl.act.lrelu(x, 0.01)
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)

    nx = Input(x_shape)
    # n = Conv2d(df_dim, (4,4), (2,2), act=lrelu, padding='SAME', W_init=w_init)(nx)
    # n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
    n = Conv2d(df_dim, (4, 4), (2, 1), act=None, padding='SAME', W_init=w_init, b_init=None)(nx)
    n = LayerNorm(gamma_init=g_init, act=lrelu)(n)

    # n = Conv2d(df_dim*2, (4,4), (2,2), act=lrelu, padding='SAME', W_init=w_init)(n)
    n = Conv2d(df_dim * 2, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=None)(n)
    n = LayerNorm(gamma_init=g_init, act=lrelu)(n)
    # n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
    df_dim = df_dim * 2

    # res block*6
    for res_i in range(0, 3):
        nn = Conv2d(df_dim, (3, 3), (1, 1), act=None,
                    padding='SAME', W_init=w_init, b_init=None)(n)
        # nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
        nn = LayerNorm(gamma_init=g_init, act=lrelu)(nn)
        nn = Conv2d(df_dim, (3, 3), (1, 1), act=None,
                    padding='SAME', W_init=w_init, b_init=None)(nn)
        # nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
        nn = LayerNorm(gamma_init=g_init, act=None)(nn)
        n = Elementwise(tf.add, act=lrelu)([n, nn])

    for i in range(2, 6):
        # n = Conv2d(df_dim*2, (4,4), (2,2), act=lrelu, padding='SAME', W_init=w_init)(n)
        n = Conv2d(df_dim * 2, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=None)(n)
        n = LayerNorm(gamma_init=g_init, act=lrelu)(n)
        # n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
        df_dim = df_dim * 2

    n_rf = Conv2d(1, (3, 3), (ks, ks), padding='SAME', b_init=None, W_init=w_init)(n)
    n_rf = Flatten()(n_rf)
    n_rf = Lambda(tf.nn.sigmoid)(n_rf)

    n_label_1 = Conv2d(4, (ks, ks), (ks, ks), padding='SAME', W_init=w_init, b_init=None)(n)
    n_label_1 = Flatten()(n_label_1)
    n_label_1 = Lambda(tf.nn.softmax)(n_label_1)

    n_label_2 = Conv2d(4, (ks, ks), (ks, ks), padding='SAME', W_init=w_init, b_init=None)(n)
    n_label_2 = Flatten()(n_label_2)
    n_label_2 = Lambda(tf.nn.softmax)(n_label_2)

    n_label = Concat(-1)([n_label_1, n_label_2])

    assert n_rf.shape[-1] == 1
    assert n_label.shape[-1] == flags.t_dim
    #print(nx.shape, n_rf.shape, n_label.shape)
    return tl.models.Model(inputs=nx, outputs=[n_rf, n_label])


## deeper version
###from ICCV17 Semantic-location-gan by DH, encode 16*16 image 128*128 +deep
# def get_D(x_shape=(None, flags.im_sz, flags.im_sz, 3), txt_shape=(None, flags.t_dim)):
#    df_dim = 64  # 64 for flower, 196 for MSCOCO
#    s = flags.im_sz  # output image size [128]
#    s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
#    lrelu = lambda x: tl.act.lrelu(x, 0.2)
#    w_init = tf.random_normal_initializer(stddev=0.02)
#    g_init = tf.random_normal_initializer(1., 0.02)
#
#    nx = Input(shape=x_shape, name='imagein')
#    n = Conv2d(df_dim, (3, 3), (2, 2), act=lrelu,
#               padding='SAME', W_init=w_init, name='c0')(nx)
#
#    n = Conv2d(df_dim * 2, (4, 4), (2, 2), act=None,
#                padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(df_dim * 4, (4, 4), (2, 2), act=None,
#                padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(df_dim * 8, (4, 4), (2, 2), act=None,
#                padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, gamma_init=g_init)(n)
#
#    #res block*3
#    for res_i in range(0, 3):
#        nn = Conv2d(df_dim*2, (1,1), (1,1), act=None,
#                    padding='VALID', W_init=w_init, b_init=None)(n)
#        nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
#        nn = Conv2d(df_dim*2, (3,3), (1,1), act=None,
#                    padding='SAME', W_init=w_init, b_init=None)(nn)
#        nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
#        nn = Conv2d(df_dim*8, (3,3), (1,1), act=None,
#                    padding='SAME', W_init=w_init, b_init=None)(nn)
#        nn = BatchNorm2d(decay=0.9, gamma_init=g_init)(nn)
#        n = Elementwise(tf.add, act=lrelu)([n, nn])
#
#    # if t_txt is not None:
#    nt = Input(shape=txt_shape, name='txtin')
#    # ntt = Dense(n_units=flags.t_dim,
#    #             act=lrelu, W_init=w_init)(nt)
#    ntt = ExpandDims(1)(nt)
#    ntt = ExpandDims(1)(ntt)
#    ntt = Tile([1, 8, 8, 1])(ntt)  # (1, 4, 4, 11)
#    n = Concat(concat_dim=3)([n, ntt])
#    # n = Conv2d(df_dim * 8, (1, 1), (1, 1),
#    #            padding='VALID', W_init=w_init, b_init=None)(n)
#    # n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(df_dim * 8, (3, 3), (1, 1),
#               padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(df_dim * 8, (3, 3), (1, 1),
#                padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(df_dim * 8, (3, 3), (1, 1),
#                padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(1, (s16, s16), (s16, s16), padding='SAME', W_init=w_init)(n)
#    # 1 x 1 x 1
#    n = Flatten()(n)
#    # n = Lambda(tf.nn.sigmoid)(n)
#    assert n.shape[-1] == 1
#
#    return tl.models.Model(inputs=[nx, nt], outputs=n)

#
# # normal version
# ##from ICCV17 Semantic-location-gan by DH, encode 16*16 image 128*128
# # concat text and image at the beginning
# def get_D(x_shape=(None, flags.im_sz, flags.im_sz, 3), txt_shape=(None, flags.t_dim)):
#     df_dim = 64  # 64 for flower, 196 for MSCOCO
#     s = flags.im_sz  # output image size [128]
#     s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
#     lrelu = lambda x: tl.act.lrelu(x, 0.2)
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     g_init = tf.random_normal_initializer(1., 0.02)
#
#     nx = Input(shape=x_shape, name='imagein')
#     nt = Input(shape=txt_shape, name='txtin')
#     ntt = ExpandDims(1)(nt)
#     ntt = ExpandDims(1)(ntt)
#     ntt = Tile([1, flags.im_sz, flags.im_sz, 1])(ntt)  # (1, 128, 128, 5)
#     n = Concat(concat_dim=3)([nx, ntt])  # (1, 128, 128, 8)
#
#     n = Conv2d(df_dim, (3, 3), (2, 2), act=lrelu,   # 64
#                padding='SAME', W_init=w_init, name='c0')(n)
#     n = Conv2d(df_dim * 2, (4, 4), (2, 2), act=None,    # 32
#                padding='SAME', W_init=w_init, b_init=None)(n)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#     n = Conv2d(df_dim * 4, (4, 4), (2, 2), act=None,    # 16
#                padding='SAME', W_init=w_init, b_init=None)(n)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#     n = Conv2d(df_dim * 8, (4, 4), (2, 2), act=None,    # 8
#                padding='SAME', W_init=w_init, b_init=None)(n)
#     n = BatchNorm2d(decay=0.9, gamma_init=g_init)(n)
#
#     # res block*3
#     for res_i in range(0, 3):
#         nn = Conv2d(df_dim * 8, (1, 1), (1, 1), act=None,
#                     padding='VALID', W_init=w_init, b_init=None)(n)
#         nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
#         nn = Conv2d(df_dim * 8, (3, 3), (1, 1), act=None,
#                     padding='SAME', W_init=w_init, b_init=None)(nn)
#         nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
#         nn = Conv2d(df_dim * 8, (3, 3), (1, 1), act=None,
#                       padding='SAME', W_init=w_init, b_init=None)(nn)
#         nn = BatchNorm2d(decay=0.9, gamma_init=g_init)(nn)
#         n = Elementwise(tf.add, act=lrelu)([n, nn])
#
#     n = Conv2d(1, (s16, s16), (s16, s16), padding='SAME', W_init=w_init)(n)  # 1
#     # 1 x 1 x 1
#     n = Flatten()(n)
#     # n = Lambda(tf.nn.sigmoid)(n)
#     assert n.shape[-1] == 1
#
#     return tl.models.Model(inputs=[nx, nt], outputs=n)

## 64*64 deep
###from ICCV17 Semantic-location-gan by DH, encode 16*16
# def get_D(x_shape=(None, flags.im_sz, flags.im_sz, 3), txt_shape=(None, flags.t_dim)):
#    df_dim = 64  # 64 for flower, 196 for MSCOCO
#    s = flags.im_sz  # output image size [64]
#    s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
#    lrelu = lambda x: tl.act.lrelu(x, 0.2)
#    w_init = tf.random_normal_initializer(stddev=0.02)
#    g_init = tf.random_normal_initializer(1., 0.02)
#
#    nx = Input(shape=x_shape, name='imagein')
#    n = Conv2d(df_dim, (3, 3), (2, 2), act=lrelu,
#               padding='SAME', W_init=w_init, name='c0')(nx)
#
#    n = Conv2d(df_dim * 2, (4, 4), (2, 2), act=None,
#                padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(df_dim * 4, (4, 4), (2, 2), act=None,
#                padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(df_dim * 8, (4, 4), (2, 2), act=None,
#                padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, gamma_init=g_init)(n)
#
#    #res block*3
#    for res_i in range(0, 3):
#        nn = Conv2d(df_dim*2, (1,1), (1,1), act=None,
#                    padding='VALID', W_init=w_init, b_init=None)(n)
#        nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
#        nn = Conv2d(df_dim*2, (3,3), (1,1), act=None,
#                    padding='SAME', W_init=w_init, b_init=None)(nn)
#        nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
#        nn = Conv2d(df_dim*8, (3,3), (1,1), act=None,
#                    padding='SAME', W_init=w_init, b_init=None)(nn)
#        nn = BatchNorm2d(decay=0.9, gamma_init=g_init)(nn)
#        n = Elementwise(tf.add, act=lrelu)([n, nn])
#
#    # if t_txt is not None:
#    nt = Input(shape=txt_shape, name='txtin')
#    # ntt = Dense(n_units=flags.t_dim,
#    #             act=lrelu, W_init=w_init)(nt)
#    ntt = ExpandDims(1)(nt)
#    ntt = ExpandDims(1)(ntt)
#    ntt = Tile([1, 4, 4, 1])(ntt)  # (1, 4, 4, 11)
#    n = Concat(concat_dim=3)([n, ntt])
#    # n = Conv2d(df_dim * 8, (1, 1), (1, 1),
#    #            padding='VALID', W_init=w_init, b_init=None)(n)
#    # n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(df_dim * 8, (3, 3), (1, 1),
#                padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(df_dim * 8, (3, 3), (1, 1),
#                padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(df_dim * 8, (3, 3), (1, 1),
#                padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(1, (s16, s16), (s16, s16), padding='SAME', W_init=w_init)(n)
#    # 1 x 1 x 1
#    n = Flatten()(n)
#    # n = Lambda(tf.nn.sigmoid)(n)
#    assert n.shape[-1] == 1
#
#    return tl.models.Model(inputs=[nx, nt], outputs=n)

## 64*64 no resblock
###from ICCV17 Semantic-location-gan by DH, encode 16*16
# def get_D(x_shape=(None, flags.im_sz, flags.im_sz, 3), txt_shape=(None, flags.t_dim)):
#    df_dim = 64  # 64 for flower, 196 for MSCOCO
#    s = flags.im_sz  # output image size [64]
#    s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
#    lrelu = lambda x: tl.act.lrelu(x, 0.01)
#    w_init = tf.random_normal_initializer(stddev=0.02)
#    g_init = tf.random_normal_initializer(1., 0.02)
#
#    nx = Input(shape=x_shape, name='imagein')
#    n = Conv2d(df_dim, (3, 3), (2, 2), act=lrelu,
#               padding='SAME', W_init=w_init, name='c0')(nx)
#
#    n = Conv2d(df_dim * 2, (3, 3), (2, 2), act=None,
#               padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(df_dim * 4, (3, 3), (2, 2), act=None,
#               padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#
#    # if t_txt is not None:
#    nt = Input(shape=txt_shape, name='txtin')
#    # ntt = Dense(n_units=flags.t_dim,
#    #             act=lrelu, W_init=w_init)(nt)
#    ntt = ExpandDims(1)(nt)
#    ntt = ExpandDims(1)(ntt)
#    ntt = Tile([1, 8, 8, 1])(ntt)  # (1, 8, 8, 11)
#    n = Concat(concat_dim=3)([n, ntt])
#    # n = Conv2d(df_dim * 8, (1, 1), (1, 1),
#    #            padding='VALID', W_init=w_init, b_init=None)(n)
#    # n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(df_dim * 8, (3, 3), (1, 1),
#               padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(df_dim * 8, (3, 3), (2, 2),
#               padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(df_dim * 8, (3, 3), (1, 1),
#               padding='SAME', W_init=w_init, b_init=None)(n)
#    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
#    n = Conv2d(1, (s16, s16), (s16, s16), padding='SAME', W_init=w_init)(n)
#    # 1 x 1 x 1
#    n = Flatten()(n)
#    # n = Lambda(tf.nn.sigmoid)(n)
#    assert n.shape[-1] == 1
#
#    return tl.models.Model(inputs=[nx, nt], outputs=n)


##from ICCV17 Semantic-location-gan by DH, encode 64*64
# def get_D(x_shape=(None, 256, 256, 3), txt_shape=(None, flags.t_dim)):
#    df_dim = 64  # 64 for flower, 196 for MSCOCO
#    s = 256 # output image size [64]
#    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
#    lrelu = lambda x: tl.act.lrelu(x, 0.2)
#    w_init = tf.random_normal_initializer(stddev=0.02)
#    g_init = tf.random_normal_initializer(1., 0.02)
#
#    nx = Input(shape=x_shape, name='imagein')
#    n = Conv2d(df_dim, (4, 4), (2, 2), act=tf.nn.relu,
#               padding='SAME', W_init=w_init, name='c0')(nx)
#
#    n = Conv2d(df_dim*2, (4, 4), (2, 2), act=None,
#                          padding='SAME', W_init=w_init, b_init=None, name='c1')(n)
#    n = InstanceNorm2d(act=lrelu, gamma_init=g_init, name='b1')(n)
#    n = Conv2d(df_dim*4, (4, 4), (2, 2), act=None,
#                          padding='SAME', W_init=w_init, b_init=None, name='c2')(n)
#    n = InstanceNorm2d(act=lrelu, gamma_init=g_init, name='b2')(n)
#    n = Conv2d(df_dim*8, (4, 4), (2, 2), act=None,
#                          padding='SAME', W_init=w_init, b_init=None, name='c3')(n)
#    n = InstanceNorm2d(act=lrelu, gamma_init=g_init, name='b3')(n)
#
#    # if t_txt is not None:
#    nt = Input(shape=txt_shape, name='txtin')
#    ntt = Dense(n_units=flags.t_dim,
#                           act=tf.nn.relu, W_init=w_init, name='txtdense')(nt)
#    ntt = ExpandDims(1, name='txtexpanddim1')(ntt)
#    ntt = ExpandDims(1, name='txtexpanddim2')(ntt)
#    ntt = Tile([1, 16, 16, 1], name='txttile')(ntt)
#    n = Concat(concat_dim=3, name='txtconcat')([n, ntt])
#    # 243 (ndf*8 + 128 or 256) x 16 x 16
#    n = Conv2d(df_dim*8, (1, 1), (1, 1),
#                          padding='VALID', W_init=w_init, b_init=None, name='txtc1')(n)
#    n = InstanceNorm2d(act=lrelu, gamma_init=g_init, name='txtb1')(n)
#
#    n = Conv2d(1, (s16, s16), (s16, s16), padding='VALID', W_init=w_init, name='o')(n)
#    # 1 x 1 x 1
#    n = Flatten()(n)
#    #n = Lambda(tf.nn.sigmoid)(n)
#    return tl.models.Model(inputs=[nx, nt], outputs=n)

##discriminator: by DH
# def get_D(x_shape=(None, 256, 256, 3), t_shape=(None, flags.t_dim)):
#    """ 256x256 + (txt) -> real fake 40m params(too big) """
#    w_init = tf.random_normal_initializer(stddev=0.02)
#    b_init = None # tf.constant_initializer(value=0.0)
#    gamma_init=tf.random_normal_initializer(1., 0.02)
#    df_dim = 64
#    lrelu = lambda x: tl.act.lrelu(x, 0.2)
#
#    net_in = Input(x_shape)
#
#    net_h0 = Conv2d(df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init)(net_in)
#    net_h1 = Conv2d(df_dim*2, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init)(net_h0)
#    net_h1 = InstanceNorm2d(act=lrelu, gamma_init=gamma_init)(net_h1)
#    net_h2 = Conv2d(df_dim*4, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init)(net_h1)
#    net_h2 = InstanceNorm2d(act=lrelu, gamma_init=gamma_init)(net_h2)
#    net_h3 = Conv2d(df_dim*8, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init)(net_h2)
#    net_h3 = InstanceNorm2d(act=lrelu, gamma_init=gamma_init)(net_h3)
#    net_h4 = Conv2d(df_dim*16, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, )(net_h3)
#    net_h4 = InstanceNorm2d(act=lrelu, gamma_init=gamma_init)(net_h4)
#    net_h5 = Conv2d(df_dim*32, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, )(net_h4)
#    net_h5 = InstanceNorm2d(act=lrelu, gamma_init=gamma_init)(net_h5)
#    net_h6 = Conv2d(df_dim*16, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, )(net_h5)
#    net_h6 = InstanceNorm2d(act=lrelu, gamma_init=gamma_init)(net_h6)
#    net_h7 = Conv2d(df_dim*8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, )(net_h6)
#    net_h7 = InstanceNorm2d(gamma_init=gamma_init)(net_h7)
#
#    net = Conv2d(df_dim*2, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init)(net_h7)
#    net = InstanceNorm2d(act=lrelu, gamma_init=gamma_init)(net)
#    net = Conv2d(df_dim*2, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init)(net)
#    net = InstanceNorm2d(act=lrelu, gamma_init=gamma_init)(net)
#    net = Conv2d(df_dim*8, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init)(net)
#    net = InstanceNorm2d(gamma_init=gamma_init)(net)
#
#    net_h8 = Elementwise(tf.add)([net_h7, net])
#    net_h8 = Lambda(lrelu)(net_h8)
#
#    #if t_shape is not None:
#    net_tin = Input(t_shape)
#    net_txt = Dense(n_units=flags.t_dim, act=lrelu, W_init=w_init, b_init=None)(net_tin)
#    net_txt = ExpandDims(1)(net_txt)
#    net_txt = ExpandDims(1)(net_txt)
#    net_txt = Tile([1, 4, 4, 1])(net_txt)
#    net_h8_concat = Concat(3)([net_h8, net_txt])
#    net_h8 = Conv2d(df_dim*8, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=b_init)(net_h8_concat)
#    net_h8 = InstanceNorm2d(act=lrelu, gamma_init=gamma_init)(net_h8)
#
#    net_ho = Flatten()(net_h8)
#    net_ho = Dense(n_units=1, act=tf.identity, W_init = w_init)(net_ho)
#    #net_ho = Lambda(tf.nn.sigmoid, net_ho)
#
#    M = Model(inputs=[net_in, net_tin], outputs=net_ho)
#    return M

def count_weights(model):
    n_weights = 0
    for i, w in enumerate(model.all_weights):
        n = 1
        # for s in p.eval().shape:
        for s in w.get_shape():
            try:
                s = int(s)
            except:
                s = 1
            if s:
                n = n * s
        n_weights = n_weights + n
    print("num of weights (parameters) %d" % n_weights)
    return n_weights


if __name__ == '__main__':
    Ea = get_Ea()
    Ea.eval()
    print("Ea:")
    count_weights(Ea)

    Ec = get_Ec()
    Ec.eval()
    print("Ec:")
    count_weights(Ec)

    D = get_D()
    D.eval()
    print("D:")
    count_weights(D)

    G = get_G()
    G.eval()
    print("G:")
    count_weights(G)

#     appearance = np.random.normal(loc=0.0, scale=1.0, size=[flags.batch_size, flags.z_dim]).astype(np.float32)
#     test_Xa_content = np.random.normal(loc=0.0, scale=1.0, size=[flags.batch_size, flags.c_shape]).astype(np.float32)
#     np.random.normal(loc=0.0, scale=1.0, size=[flags.batch_size, flags.t_dim]).astype(np.float32)
#     # phi_tr1 = RNN(sample_sentence)
#     # phi_test_tr1_aug, m_fake_test_Xa, s_fake_test__Xa = CA(phi_tr1, disable_aug=0)
#     fake_test_Xa = G([appearance, test_Xa_content, sample_sentence])
