# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:23:57 2019
@author: Yhq
"""

import tensorlayer as tl
import tensorflow as tf
import numpy as np
import random
import time
import multiprocessing


class FLAGS(object):
    def __init__(self):
        self.dataset = 'DukeMTMC' # 'celebA'
        
        self.z_dim = 16 # dimension of appearance vector
        if self.dataset == 'DukeMTMC':
            self.im_h = 128 # size of image
            self.im_w = 64
            self.im_sz = 128
        elif self.dataset == 'celebA':
            self.im_sz = 128
        
        ## RNN architecture
        self.vocab_size = 8000
        self.word_embedding_size = 256 #20
        if self.dataset == 'DukeMTMC':
            self.c_shape = [32, 16, 256]
            self.t_dim = 8
        elif self.dataset == 'celebA':
            self.c_shape = [32, 32, 256] # shape of content tensor 16*16*256 for 128 or 64*64*256 for 256
            self.t_dim = 5
        self.t_dense_dim = 8
        
        
        ## train GAN
        self.batch_size = 32  # training batch size
        self.n_epoch = 400 # training epochs
        self.lr_init = 0.0001
        self.lr_decay_every = 100
        self.lr_decay_rate = 0.5
        self.critic_n = 3
        self.beta_1 = 0.5
        self.shuffle_buffer_size = 128
        self.shuffle_buffer_size = 128
        self.lambda_l1_cc_loss = 10
        self.lambda_l1_recon_loss = 1
        self.lambda_l1_app_loss = 5
        
        self.lambda_d_logit_loss_real = 1
        self.lambda_d_logit_loss_fake = 0.5
        self.lambda_d_logit_loss_e_fake = 0.5
        self.lambda_d_label_loss = 1

        self.lambda_g_logit_loss_fake = 1
        self.lambda_g_logit_loss_e_fake = 1
        self.lambda_g_label_loss = 1
        self.lambda_g_e_label_loss = 1
        self.lambda_gp = 10

        ## pretrain RNN
        self.batch_size_txt = 64  # training batch size
        self.n_epoch_txt = 50 # training epochs
        self.lr_init_txt = 0.0002
        self.lr_decay_every_txt = 50
        self.lr_decay_rate_txt = 0.5
        self.alpha = 0.2

        self.save_every_step = 200
        self.print_every_step = 100
        self.save_weight_every_step = 200 # how many epochs to save model and test
        self.save_weight_every_epoch = 5
        self.print_every_step = 1 # how many steps to print training info
        self.model_dir = 'models6' # folder name to save models
        self.sample_dir = 'samples6' # folder name to save visualized results
        self.prefix = '201910182300'


flags = FLAGS()
tl.files.exists_or_mkdir(flags.model_dir+'/'+flags.prefix, verbose=False)
tl.files.exists_or_mkdir(flags.sample_dir+'/'+flags.prefix, verbose=False)
tl.files.exists_or_mkdir(flags.sample_dir+'/'+flags.prefix+'/test', verbose=False)

