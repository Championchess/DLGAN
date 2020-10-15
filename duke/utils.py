# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:58:44 2019

@author: Yhq
"""

import numpy as np

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