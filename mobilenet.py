# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:07:06 2019

@author: shirhe-lyh


Reference: 
    https://github.com/tensorflow/models/blob/master/research/slim/nets/
        mobilenet/mobilenet.py
"""

import collections
import torch

import conv_blocks


_Op = collections.namedtuple('Op', ['op', 'params', 'multiplier_func'])


def half_depth_multiplier(params,
                          multiplier,
                          mode='in',
                          divisible_by=8,
                          min_depth=8,
                          **unused_kwargs):
    in_channels = params.get('in_channels', None)
    if in_channels and mode == 'in':
        params['in_channels'] = conv_blocks._make_divisible(
            in_channels * multiplier, divisible_by, min_depth)
    out_channels = params.get('out_channels', None)
    if out_channels and mode == 'out':
        params['out_channels'] = conv_blocks._make_divisible(
            out_channels * multiplier, divisible_by, min_depth)


def depth_multiplier(params,
                     multiplier,
                     divisible_by=8,
                     min_depth=8,
                     **unused_kwargs):
    in_channels = params.get('in_channels', None)
    if in_channels and in_channels >= min_depth:
        params['in_channels'] = conv_blocks._make_divisible(
            in_channels * multiplier, divisible_by, min_depth)
    out_channels = params.get('out_channels', None)
    if out_channels:
        params['out_channels'] = conv_blocks._make_divisible(
            out_channels * multiplier, divisible_by, min_depth)


def op(opfunc, multiplier_func=depth_multiplier, **params):
    multiplier = params.pop('multiplier_transform', multiplier_func)
    return _Op(opfunc, params=params, multiplier_func=multiplier)


class MobileNet(torch.nn.Module):
    """MobileNet base network."""
    
    def __init__(self,
                 conv_defs,
                 num_classes=1001,
                 global_pool=True,
                 multiplier=1.0,
                 final_endpoint=None,
                 output_stride=None,
                 use_explicit_padding=False,
                 scope=None,
                 model_name='large'):
        """Constructor.
        
        Args:
            conv_defs: A list of op(...) layers specifying the net architecture.
            num_classes: Number of classes. None or a positive integer.
            global_pool: A boolean.
            multiplier: Float multiplier for the depth (number of channels)
                for all convolution ops. The value must be greater than zero. 
                Typical usage will be to set this value in (0, 1) to reduce 
                the number of parameters or computation cost of the model.
            final_endpoint: The name of last layer, for early termination for
                for V1-based networks: last layer is "layer_14", for 
                V2: "layer_20".
            output_stride: An integer that specifies the requested ratio of 
                input to output spatial resolution. If not None, then we 
                invoke atrous convolution if necessary to prevent the network 
                from reducing the spatial resolution of the activation maps. 
                Allowed values are 1 or any even number, excluding zero. 
                Typical values are 8 (accurate fully convolutional mode), 16
                (fast fully convolutional mode), and 32 (classification mode).
      
                NOTE- output_stride relies on all consequent operators to 
                support dilated operators via "rate" parameter. This might 
                require wrapping non-conv operators to operate properly.
    
            use_explicit_padding: Use 'VALID' padding for convolutions, but 
                prepad inputs so that the output dimensions are the same as 
                if 'SAME' padding were used.
            scope: Optional variable scope.
            model_name: Name of the MobileNet V3 model, one of ['large', 
                'large_minimalistic', 'small', 'small_minimalistic', 
                'edgeTPU'].
            
        Raises:
            ValueError: multiplier <=0, or the target output_stride is not 
                allowed.
        """
        super(MobileNet, self).__init__()
        
        if multiplier <= 0:
            raise ValueError('`multiplier` is not greater than zero.')
            
        if output_stride is not None:
            if output_stride == 0 or (output_stride > 1 and output_stride % 2):
                raise ValueError('`output_stride` must be None, 1 or a '
                                 'multiple of 2.')
                
        self._scope = scope
        self._model_name = model_name
                
        # The current_stride variable keeps track of the output stride of the
        # activations, i.e., the running product of convolution strides up to 
        # the current network layer. This allows us to invoke atrous 
        # convolution whenever applying the next convolution would result in 
        # the activations having output stride larger than the target 
        # output_stride.
        current_stride = 1
        
        # The atrous convolution rate parameter.
        rate = 1
        
        layers = []
        for i, opdef in enumerate(conv_defs['spec']):
            params = dict(opdef.params)
            opdef.multiplier_func(params, multiplier)
            stride = params.get('stride', 1)
            if output_stride is not None and current_stride == output_stride:
                # If we have reached the target output_stride, then we need to 
                # employ atrous convolution with stride=1 and multiply the 
                # atrous rate by the current unit's stride for use in 
                # subsequent layers.
                layer_stride = 1
                layer_rate = rate
                rate *= stride
            else:
                layer_stride = stride
                layer_rate = 1
                current_stride *= stride
            # Update params.
            params['stride'] = layer_stride
            # Only insert rate to params if rate > 1 and kernel size is 
            # not [1, 1].
            if layer_rate > 1:
                if tuple(params.get('kernel_size', [])) != (1, 1):
                    # We will apply atrous rate in the following cases:
                    # 1) When kernel_size is not in params, the operation then 
                    # uses default kernel size 3x3.
                    # 2) When kernel_size is in params, and if the kernel_size 
                    # is not equal to (1, 1) (there is no need to apply atrous
                    # convolution to any 1x1 convolution).
                    params['dilation'] = layer_rate
            # Set padding
            if use_explicit_padding:
                if 'kernel_size' in params:
                    # TODO: do something
                    pass
                else:
                    params['use_explicit_padding'] = True
                    
            try:
                layers.append(opdef.op(**params))
            except Exception:
                print('Failed to create op: %i: %r params: %r' % (
                    i, opdef, params))
                raise
                
        # Global pooling
        global_pool = global_pool or num_classes
        if global_pool:
            layers += [torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            
        # Prediction
        self._num_classes = num_classes
        if self._num_classes:
            in_channels = params.get('out_channels', 1)
            layers += [torch.nn.Dropout2d(inplace=True),
                       torch.nn.Conv2d(in_channels=in_channels, 
                                       out_channels=num_classes,
                                       kernel_size=1)]
            
        self._layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        net = self._layers(x)
        if self._num_classes:
            net = net.view(-1, self._num_classes)
        return net
    
    @property
    def scope(self):
        return self._scope
    
    @property
    def model_name(self):
        return self._model_name