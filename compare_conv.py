# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:44:31 2019

@author: shirhe-lyh
"""

import numpy as np
import tensorflow as tf
import torch

tf.enable_eager_execution()

np.random.seed(123)
tf.set_random_seed(123)
torch.manual_seed(123)

h = 224
w = 224
k = 5
s = 2
p = k // 2 if s == 1 else 0


x_np = np.random.random((1, h, w, 3))
x_tf = tf.constant(x_np)
x_pth = torch.from_numpy(x_np.transpose(0, 3, 1, 2))


def pad(x, kernel_size=3, dilation=1):
    """For stride = 2 or stride = 3"""
    pad_total = dilation * (kernel_size - 1) - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    x_padded = torch.nn.functional.pad(
        x, pad=(pad_beg, pad_end, pad_beg, pad_end))
    return x_padded


conv_tf = tf.layers.Conv2D(filters=16, 
                           padding='SAME',
                           kernel_size=k,
                           strides=(s, s))

# Tensorflow prediction
with tf.GradientTape(persistent=True) as t:
    t.watch(x_tf)
    y_tf = conv_tf(x_tf).numpy()
    print('Shape: ', y_tf.shape)
    
    
conv_pth = torch.nn.Conv2d(in_channels=3,
                           out_channels=16,
                           kernel_size=k,
                           stride=s,
                           padding=p)

# Reset parameters
weights_tf, biases_tf = conv_tf.get_weights()
conv_pth.weight.data = torch.tensor(weights_tf.transpose(3, 2, 0, 1))
conv_pth.bias.data = torch.tensor(biases_tf)


# Pytorch prediction
conv_pth.eval()
with torch.no_grad():
    if s > 1:
        x_pth = pad(x_pth, kernel_size=k)
    y_pth = conv_pth(x_pth)
    y_pth = y_pth.numpy().transpose(0, 2, 3, 1)
    print('Shape: ', y_pth.shape)
    
    
# Compare results
print('y_tf: ')
print(y_tf[:, h//s-1, 0, :])
print('y_pth: ')
print(y_pth[:, h//s-1, 0, :])  