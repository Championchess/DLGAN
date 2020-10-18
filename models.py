# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import (BatchNorm2d, Conv2d, Dense, Flatten, Input, DeConv2d, Lambda, \
                                LocalResponseNorm, MaxPool2d, Elementwise, InstanceNorm2d, \
                                Concat, ExpandDims, Tile, GlobalMeanPool2d, UpSampling2d, \
                                MeanPool2d, GaussianNoise, LayerNorm)
from tensorlayer.models import Model
from config import flags


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

'''
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
'''


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


# content encoder: input X, output c
# ref: DRIT Ec 128*128->32*32*256
def get_Ec_DukeMTMC(x_shape=(None, flags.im_h, flags.im_w, 3), c_shape= \
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

def get_Ec_celebA(x_shape=(None, flags.im_sz, flags.im_sz, 3), c_shape= \
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
        n = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(n)
        channel = channel * 2

    for i in range(0, 3):
        # res block
        nn = Conv2d(channel, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
        # nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
        nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
        nn = Conv2d(channel, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
        # nn = InstanceNorm2d(act=None, gamma_init=g_init)(nn)
        nn = BatchNorm2d(decay=0.9, act=None, gamma_init=g_init)(nn)
        n = Elementwise(tf.add, act=tf.nn.relu)([n, nn])

    # n = GaussianNoise(is_always=False)(n)
    M = Model(inputs=ni, outputs=n, name=name)
    return M


# appearance encoder
def get_Ea_DukeMTMC(x_shape=(None, flags.im_h, flags.im_w, 3), z_shape=(None, flags.z_dim), name=None):
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

def get_Ea_celebA(x_shape=(None, flags.im_sz, flags.im_sz, 3), z_shape=(None, flags.z_dim), name=None):
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
        n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
        n = Conv2d(ndf * (i + 1), (3, 3), (1, 1), padding='VALID', W_init=w_init, b_init=None, act=None)(
            n)  # conv3x3 in convMeanpool
        n = BatchNorm2d(decay=0.9, act=None, gamma_init=g_init)(n)
        n = MeanPool2d((2, 2), (2, 2))(n)  # meanPool2d in convMeanpool
        # shortcut part
        ns = MeanPool2d((2, 2), (2, 2))(nn)
        ns = Conv2d(ndf * (i + 1), (3, 3), (1, 1), padding='VALID', b_init=None, W_init=w_init, act=None)(ns)
        nn = Elementwise(tf.add, act=lrelu)([n, ns])

    n = GlobalMeanPool2d()(nn)

    n = Dense(flags.z_dim, W_init=w_init, act=None)(n)

    M = Model(inputs=ni, outputs=n, name=name)
    return M




# generator: input a, c, txt, output X', encode 32*32*256
# ref: DRIT+stargan DH (split the t vector)
def get_G_DukeMTMC(a_shape=(None, flags.z_dim),
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

def get_G_celebA(a_shape=(None, flags.z_dim),
          c_shape=(None, flags.c_shape[0], flags.c_shape[1], flags.c_shape[2]),
          t_shape=(None, flags.t_dim),
          # nt_params=None,
          name=None):
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)
    ndf = 256

    na = Input(a_shape)
    nc = Input(c_shape)  # (1, 16, 16, 512)
    nt = Input(t_shape)
    nat = Concat(-1)([na, nt])  # (1, 133) = (1, 128) + (1, 5)
    nat = ExpandDims(1)(nat)  # (1, 1, 133)
    nat = ExpandDims(1)(nat)  # (1, 1, 1, 133)
    nat = Tile([1, flags.c_shape[0], flags.c_shape[1], 1])(nat)  # (1, 16, 16, 133)
    n = Concat(-1)([nat, nc])  # (1, 16, 16, 645)
    nd_tmp = flags.z_dim + flags.t_dim  # 133
    ndf = ndf + nd_tmp  # 512+133=645

    # res block *6
    for i in range(6):
        nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
        # nn = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
        nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(nn)
        nn = Conv2d(ndf, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
        # nn = InstanceNorm2d(act=None, gamma_init=g_init)(nn)
        nn = BatchNorm2d(decay=0.9, act=None, gamma_init=g_init)(nn)
        n = Elementwise(tf.add, act=lrelu)([n, nn])

    ngt = ExpandDims(1)(nt)
    ngt = ExpandDims(1)(ngt)
    ngt = Tile([1, flags.c_shape[0], flags.c_shape[1], 1])(ngt)  # (1, 16, 16, 5)
    n = Concat(-1)([n, ngt])  # (1, 16, 16, 650)
    ndf += flags.t_dim
    
    for i in range(2):
        n = DeConv2d(ndf // 2, (3, 3), (2, 2), act=None, W_init=w_init, b_init=None)(n)
        #print("step"+str(i)+":"+str(n.shape))
        # n = UpSampling2d((2, 2))(n)
        # n = Conv2d(ndf//2, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
        # n = InstanceNorm2d(act=tf.nn.relu, gamma_init=g_init)(n)
        n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)

        ndf = ndf // 2

    #print(n.shape)
    # n = Concat(-1)([n, nat])
    n = Conv2d(3, (1, 1), (1, 1), act=tf.nn.tanh, W_init=w_init)(n)
    #print(n.shape)
    M = Model(inputs=[na, nc, nt], outputs=n, name=name)
    return M



# ref: stargan D image 128*128
def get_D_DukeMTMC(x_shape=(None, flags.im_h, flags.im_w, 3)):
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

def get_D_celebA(x_shape=(None, flags.im_sz, flags.im_sz, 3)):
    df_dim = 64  # 64 for flower, 196 for MSCOCO
    s = flags.im_sz  # output image size [128]
    ks = int(flags.im_sz / 64)
    lrelu = lambda x: tl.act.lrelu(x, 0.01)
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)

    nx = Input(x_shape)
    # n = Conv2d(df_dim, (4,4), (2,2), act=lrelu, padding='SAME', W_init=w_init)(nx)
    # n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=g_init)(n)
    n = Conv2d(df_dim, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=None)(nx)
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

    n_label_1 = Conv2d(3, (ks, ks), (ks, ks), padding='SAME', W_init=w_init, b_init=None)(n)
    n_label_1 = Flatten()(n_label_1)
    n_label_1 = Lambda(tf.nn.softmax)(n_label_1)

    n_label_2 = Conv2d(2, (ks, ks), (ks, ks), padding='SAME', W_init=w_init, b_init=None)(n)
    n_label_2 = Flatten()(n_label_2)
    n_label_2 = Lambda(tf.nn.softmax)(n_label_2)

    n_label = Concat(-1)([n_label_1, n_label_2])

    assert n_rf.shape[-1] == 1
    assert n_label.shape[-1] == flags.t_dim

    return tl.models.Model(inputs=nx, outputs=[n_rf, n_label])



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


'''
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
'''
