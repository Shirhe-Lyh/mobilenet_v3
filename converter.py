# -*- coding: utf-8 -*-
"""
Created on Mon Dec  12 21:01:52 2019

@author: shirhe-lyh


Convert tensorflow weights to pytorch weights for Xception models.

Reference:
    https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/
        tf_to_pytorch/convert_tf_to_pt/load_tf_weights.py
"""

import numpy as np
import tensorflow as tf
import torch


_MBV3_CONV_COUNT = {
    'large': [15, 2],
    'small': [11, 2],
    'edgeTPU': [22, 1],
}


def load_param(checkpoint_path, conversion_map, scope='MobilenetV3'):
    """Load parameters according to conversion_map.
    
    Args:
        checkpoint_path: Path to tensorflow's checkpoint file.
        conversion_map: A dictionary with format 
            {pytorch tensor in a model: checkpoint variable name}
        scope: The name scope of MobileNet V3 models.
    """
    for pth_param, tf_param_name in conversion_map.items():
        tf_param_name = str(scope) + '/' + tf_param_name
        tf_param = tf.train.load_variable(checkpoint_path, tf_param_name)
        if 'CONV' in tf_param_name.upper() and 'weights' in tf_param_name:
            tf_param = np.transpose(tf_param, (3, 2, 0, 1))
            if 'depthwise' in tf_param_name:
                tf_param = np.transpose(tf_param, (1, 0, 2, 3))
        elif tf_param_name.endswith('weights'):
            tf_param = np.transpose(tf_param)
        assert pth_param.size() == tf_param.shape, ('Dimension mismatch: ' + 
            '{} vs {}; {}'.format(pth_param.size(), tf_param.shape, 
                 tf_param_name))
        pth_param.data = torch.from_numpy(tf_param)


def convert(model, checkpoint_path):
    """Load Pytorch MobileNet V3 from TensorFlow checkpoint file.
    
    Args:
        model: The pytorch MobileNet V3 model.
        checkpoint_path: Path to tensorflow's checkpoint file.
        
    Returns:
        The pytorch MobileNet V3 model with pretrained parameters.
    """
    mbv3_conv_count = _MBV3_CONV_COUNT.get(model.model_name.split('_')[0], None)
    if mbv3_conv_count is None:
        raise ValueError('Unknown model nickname: %s' %model.nickname)
    
    conversion_map = {}
    # Root block
    conversion_map_for_root_block = {
        model._layers[0]._layers[0].weight: 
            'Conv/weights',
        model._layers[0]._layers[1].bias: 
            'Conv/BatchNorm/beta',
        model._layers[0]._layers[1].weight:
            'Conv/BatchNorm/gamma',
        model._layers[0]._layers[1].running_mean: 
            'Conv/BatchNorm/moving_mean',
        model._layers[0]._layers[1].running_var: 
            'Conv/BatchNorm/moving_variance',
    }
    conversion_map.update(conversion_map_for_root_block)
    
    # MBV3
    num_mbv3s, num_convs = mbv3_conv_count
    for i in range(num_mbv3s):
        expanded_conv_name = 'expanded_conv'
        if i > 0:
            expanded_conv_name += '_{}'.format(i)
        
        # Projection conv (split_expansion = num_ways = 1)
        layers = model._layers[1+i]._projection_fn._branches[0]
        local_scope = '{}/{}'.format(expanded_conv_name, 'project')
        conversion_map_for_mb = {
            layers[0].weight: 
                '{}/weights'.format(local_scope),
            layers[1].bias:
                '{}/BatchNorm/beta'.format(local_scope),
            layers[1].weight: 
                '{}/BatchNorm/gamma'.format(local_scope),
            layers[1].running_mean: 
                '{}/BatchNorm/moving_mean'.format(local_scope),
            layers[1].running_var: 
                '{}/BatchNorm/moving_variance'.format(local_scope),
        }
        conversion_map.update(conversion_map_for_mb)
            
        # Expansion conv
        inner_size = model._layers[1+i]._inner_size
        in_channels = model._layers[1+i]._in_channels
        if inner_size > in_channels:
            layers = model._layers[1+i]._expansion_fn._branches[0]
            local_scope = '{}/{}'.format(expanded_conv_name, 'expand')
            conversion_map_for_mb = {
                layers[0].weight: 
                    '{}/weights'.format(local_scope),
                layers[1].bias:
                    '{}/BatchNorm/beta'.format(local_scope),
                layers[1].weight: 
                    '{}/BatchNorm/gamma'.format(local_scope),
                layers[1].running_mean: 
                    '{}/BatchNorm/moving_mean'.format(local_scope),
                layers[1].running_var: 
                    '{}/BatchNorm/moving_variance'.format(local_scope),
            }
            conversion_map.update(conversion_map_for_mb)
        
        # Depthwise conv
        depthwise_location = model._layers[1+i]._depthwise_location
        if depthwise_location:
            layers = model._layers[1+i]._depthwise_func._layers
            local_scope = '{}/{}'.format(expanded_conv_name, 'depthwise')
            conversion_map_for_mb = {
                layers[0].weight: 
                    '{}/depthwise_weights'.format(local_scope),
                layers[1].bias: 
                    '{}/BatchNorm/beta'.format(local_scope),
                layers[1].weight: 
                    '{}/BatchNorm/gamma'.format(local_scope),
                layers[1].running_mean: 
                    '{}/BatchNorm/moving_mean'.format(local_scope),
                layers[1].running_var: 
                    '{}/BatchNorm/moving_variance'.format(local_scope),
            }
            conversion_map.update(conversion_map_for_mb)
                
        # Squeeze and excitation conv
        expansion_transform = model._layers[1+i]._expansion_transform
        if expansion_transform is not None:
            layers = model._layers[1+i]._expansion_transform._layers
            local_scope = '{}/{}/{}'.format(expanded_conv_name, 
                                           'squeeze_excite',
                                           'Conv')
            if len(layers) > 3:
                # mbv3_op_se
                for i in range(2):
                    if i > 0:
                        local_scope += '_{}'.format(i)
                    index = 2 * i + 1
                    conversion_map_for_mb = {
                        layers[index].weight: 
                            '{}/weights'.format(local_scope),
                        layers[index].bias: 
                            '{}/biases'.format(local_scope),
                    }
                    conversion_map.update(conversion_map_for_mb)
            else:
                # mbv3_fused
                conversion_map_for_mb = {
                    layers[0].weight: 
                        '{}/weights'.format(local_scope),
                    layers[1].bias:
                        '{}/BatchNorm/beta'.format(local_scope),
                    layers[1].weight: 
                        '{}/BatchNorm/gamma'.format(local_scope),
                    layers[1].running_mean: 
                        '{}/BatchNorm/moving_mean'.format(local_scope),
                    layers[1].running_var: 
                        '{}/BatchNorm/moving_variance'.format(local_scope),
                }
                conversion_map.update(conversion_map_for_mb)
    
    # Last Convs
    index = num_mbv3s + 1
    conversion_map_for_last_conv_block = {
        model._layers[index]._layers[0].weight: 
            'Conv_{}/weights'.format(1),
        model._layers[index]._layers[1].bias: 
            'Conv_{}/BatchNorm/beta'.format(1),
        model._layers[index]._layers[1].weight: 
            'Conv_{}/BatchNorm/gamma'.format(1),
        model._layers[index]._layers[1].running_mean: 
            'Conv_{}/BatchNorm/moving_mean'.format(1),
        model._layers[index]._layers[1].running_var: 
            'Conv_{}/BatchNorm/moving_variance'.format(1),
    }
    conversion_map.update(conversion_map_for_last_conv_block)
    if num_convs > 1:
        conversion_map_for_last_conv_block = {
            model._layers[index+2]._layers[0].weight: 
                'Conv_{}/weights'.format(2),
            model._layers[index+2]._layers[0].bias: 
                'Conv_{}/biases'.format(2),
        }
        conversion_map.update(conversion_map_for_last_conv_block)
        
    # Prediction
    conversion_map_for_prediction = {
    model._layers[-1].weight: 'Logits/Conv2d_1c_1x1/weights',
    model._layers[-1].bias: 'Logits/Conv2d_1c_1x1/biases',
    }
    conversion_map.update(conversion_map_for_prediction)
        
    # Load TensorFlow parameters into PyTorch model
    load_param(checkpoint_path, conversion_map, model.scope)