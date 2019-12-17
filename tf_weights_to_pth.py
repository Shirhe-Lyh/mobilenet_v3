# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:59:01 2019

@author: shirhe-lyh
"""

import cv2
import functools
import numpy as np
import os
import tensorflow as tf
import torch

import converter
import mobilenet_v3
import original_tf.mobilenet_v3 as tf_mobilenet_v3

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('model_name',
                    'large',
                    'The nickname of MobileNet V3 model. one of ["large", ' +
                    '"large_minimalistic", "small", "small_minimalistic", ' +
                    '"edgeTPU"].')
flags.DEFINE_float('depth_multiplier',
                   1.0,
                   'The multiplier applied to scale number of channels in ' +
                   'each layer.')
flags.DEFINE_boolean('finegrain_classification_mode',
                     True,
                     'When set to True, the model will keep the last layer '+
                     'large even for small multipliers.')
flags.DEFINE_string('tf_checkpoint_path',
                    None,
                    'Path to the tensorflow checkpoint file.')
flags.DEFINE_string('output_dir',
                    './pretrained_models',
                    'Where the converted checkpoint file is stored.')
flags.DEFINE_string('output_name',
                    'mobilenet_v3_large.pth',
                    'The name of converted checkpoint file.')
flags.DEFINE_string('image_path',
                    './test/panda.jpg',
                    'Path to a test image.')

FLAGS = flags.FLAGS


if __name__ == '__main__':
    model_name = FLAGS.model_name
    depth_multiplier = FLAGS.depth_multiplier
    finegrain_classification_mode = FLAGS.finegrain_classification_mode
    checkpoint_path = FLAGS.tf_checkpoint_path
    output_dir = FLAGS.output_dir
    output_name = FLAGS.output_name
    image_path = FLAGS.image_path
    
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(image_path):
        raise ValueError('`image_path` does not exist.')
    if not os.path.exists(checkpoint_path + '.index'):
        raise ValueError('`tf_checkpoint_path` does not exit.')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_name)
    
    image = cv2.imread(image_path)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224))
    image_center = (2.0 / 255) * image_rgb - 1.0
    image_center = image_center.astype(np.float32)
    images = np.expand_dims(image_center, axis=0)
    images_pth = np.expand_dims(np.transpose(image_center, axes=(2, 0, 1)),
                               axis=0)
    images_pth = torch.from_numpy(images_pth).to(device)
    
    # -----------------------------------
    # Tensorflow mobilenet_v3 model
    # -----------------------------------
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='inputs')
    
    # Note: arg_scope is optional for inference.
    with slim.arg_scope(tf_mobilenet_v3.training_scope(is_training=False)):
        if model_name == 'large':
            model_tf = tf_mobilenet_v3.large
        elif model_name == 'large_minimalistic':
            model_tf = tf_mobilenet_v3.large_minimalistic
        elif model_name == 'small':
            model_tf = tf_mobilenet_v3.small
        elif model_name == 'small_minimalistic':
            model_tf = tf_mobilenet_v3.small_minimalistic
        elif model_name == 'edgeTPU':
            model_tf = tf_mobilenet_v3.edge_tpu
        else:
            raise ValueError('Unknown `model_name`: %s' %model_name)
            
        _, endpoints = model_tf(inputs, depth_multiplier=depth_multiplier,
            finegrain_classification_mode=finegrain_classification_mode)
            
    predictions = endpoints.get('Predictions')
    classes = tf.argmax(predictions, axis=1)
    
    init = tf.global_variables_initializer()
    
    var_list=slim.get_model_variables()
    saver = tf.train.Saver(var_list=var_list)

    with tf.Session() as sess:
        sess.run(init)
        
        # Load tensorflow pretrained paremeters
        saver.restore(sess, checkpoint_path)
        
        logits, labels = sess.run([predictions, classes],
                                  feed_dict={inputs: images})
        print('---------------------')
        print('TensorFlow predicion:')
        print('Label: ', labels)
        print('Top5 : ', np.argsort(logits)[:, -5:][0][::-1])
        
    
    # -----------------------------------
    # Pytorch mobilenet_v3 model
    # -----------------------------------
    if model_name == 'large':
        model = mobilenet_v3.large
    elif model_name == 'large_minimalistic':
        model = mobilenet_v3.large_minimalistic
    elif model_name == 'small':
        model = mobilenet_v3.small
    elif model_name == 'small_minimalistic':
        model = mobilenet_v3.small_minimalistic
    else:
        model = mobilenet_v3.edge_tpu
    model_pth = functools.partial(
        model, 
        depth_multiplier=depth_multiplier,
        finegrain_classification_mode=finegrain_classification_mode)
    model_pth = model_pth().to(device)
        
    # Convert tensorflow pretrained weights to pytorch weights
    converter.convert(model_pth, checkpoint_path)
    
    model_pth.eval()
    with torch.no_grad():
        logits_pth = torch.nn.functional.softmax(model_pth(images_pth), dim=1)
        logits_pth = logits_pth.data.cpu().numpy()
        labels_pth = np.argmax(logits_pth, axis=1)
        print('---------------------')
        print('PyTorch prediction:')
        print('Label: ', labels_pth)
        print('Top5 : ', np.argsort(logits_pth)[:, -5:][0][::-1])
        
    # Save convterted pretrained parameters
    if not os.path.exists(output_path):
        torch.save(model_pth.state_dict(), output_path)
        print('Save model to: ', output_path)
    
    
    # ---------------------Test---------------------
    # Load pretrained parameters for Pytorch mobilenet_v3 model
    print('---------------------')
    print('Define Pytorch mobilenet_v3 model with pretrained parameters.')
    model_pretrained = functools.partial(
        model,
        num_classes=1001,
        depth_multiplier=depth_multiplier,
        finegrain_classification_mode=finegrain_classification_mode,
        pretrained=True,
        checkpoint_path=output_path)
    model_pretrained = model_pretrained().to(device)
    
    model_pretrained.eval()
    with torch.no_grad():
        logits_pth = torch.nn.functional.softmax(
            model_pretrained(images_pth), dim=1)
        logits_pth = logits_pth.data.cpu().numpy()
        labels_pth = np.argmax(logits_pth, axis=1)
        print('---------------------')
        print('PyTorch (pretrained) prediction:')
        print('Label: ', labels_pth)
        print('Top5 : ', np.argsort(logits_pth)[:, -5:][0][::-1])