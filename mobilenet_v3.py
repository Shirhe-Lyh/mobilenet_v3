# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 19:07:52 2019

@author: shirhe-lyh


Reference:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/
        mobilenet/mobilenet_v3.py
"""

import copy
import functools
import os
import torch

import conv_blocks
import mobilenet as lib


class HardSigmoid(torch.nn.Module):
    """Hard sigmoid activation function."""
    
    def __init__(self,
                 inplace=True):
        """Construcor."""
        super(HardSigmoid, self).__init__()
        
        self._relu6 = torch.nn.ReLU6(inplace=inplace)
        
    def forward(self, x):
        return self._relu6(x + 3.) * 0.16667
    
    
class HardSwish(torch.nn.Module):
    """Hard Swish activation function."""
    
    def __init__(self,
                 inplace=True):
        """Construcor."""
        super(HardSwish, self).__init__()
        
        self._relu6 = torch.nn.ReLU6(inplace=inplace)
        
    def forward(self, x):
        return x * self._relu6(x + 3.) / 6.
    
    
class Conv2d(torch.nn.Module):
    """Conv2d with batch normalization and activation function."""
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1, 
                 bias=False,
                 batch_norm=True,
                 activation_fn=torch.nn.ReLU(inplace=True)):
        """Constructor."""
        super(Conv2d, self).__init__()
        
        self._without_padding = stride == 1
        if not self._without_padding:
            padding = 0
            self._kernel_size = kernel_size
            self._dilation = dilation
        layers = [torch.nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=bias,
                                  groups=groups)]
        if batch_norm:
            layers += [torch.nn.BatchNorm2d(num_features=out_channels,
                                            **conv_blocks._BATCH_NORM_PARAMS)]
        if activation_fn is not None:
            layers += [activation_fn]
            
        self._layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        if not self._without_padding:
            x = conv_blocks._same_padding(x, self._kernel_size, self._dilation)
        return self._layers(x)
    
    
class AdaptiveAvgPool2d(torch.nn.Module):
    """torch.nn.AdaptiveAvgPool2d with unused arguments."""
    
    def __init__(self, output_size, **unused_kwargs):
        """Constructor."""
        super(AdaptiveAvgPool2d, self).__init__()
        
        self._adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(output_size)
        
    def forward(self, x):
        return self._adaptive_avg_pool2d(x)


# Squeeze Excite with all parameters filled-in, we use hard-sigmoid
# for gating function and relu for inner activation function.
SqueezeExcite = functools.partial(
    conv_blocks.SqueezeExcitation, squeeze_factor=4,
    inner_activation_fn=torch.nn.ReLU(inplace=True),
    gating_fn=HardSigmoid())
    
    
def mbv3_op(ef, m, n, k, s=1, act=torch.nn.ReLU(inplace=True), se=None, 
            **kwargs):
    """Defines a single MobileNet V3 convolution block.
    
    Args:
        ef: Expansion factor.
        m: Number of input filters.
        n: Number of output filters.
        k: Stride of depthwise (kernel size).
        s: Stride.
        act: Activation function in inner layers.
        se: Squeeze excite function.
        **kwargs: Passed to ExpandedConv.
        
    Returns:
        An object (lib._Op) for inserting in conv_def, representing this
        operation.
    """
    return lib.op(
        conv_blocks.ExpandedConv,
        in_channels=m,
        out_channels=n,
        expansion_size=conv_blocks.expand_input_by_factor(ef),
        kernel_size=(k, k),
        stride=s,
        inner_activation_fn=act,
        expansion_transform=se,
        **kwargs)
    

def mbv3_fused(ef, m, n, k, s=1, **kwargs):
    """Defines a single Mobilenet V3 convolution block.
    
    Args:
        ef: expansion factor
        n: number of output channels
        k: stride of depthwise
        s: stride
        **kwargs: will be passed to mbv3_op
        
    Returns:
        An object (lib._Op) for inserting in conv_def, representing this 
        operation.
    """
    expansion_fn = functools.partial(Conv2d, kernel_size=k, stride=s,
                                     padding=k//2)
    return mbv3_op(
        ef,
        m,
        n,
        k=1,
        s=s,
        depthwise_location=None,
        expansion_fn=expansion_fn,
        **kwargs)
    
    
mbv3_op_se = functools.partial(mbv3_op, se=SqueezeExcite)


V3_LARGE = dict(
    spec=([
        # stage 1
        lib.op(Conv2d, stride=2, in_channels=3, out_channels=16,
               kernel_size=(3, 3), activation_fn=HardSwish()),
        mbv3_op(ef=1, m=16, n=16, k=3),
        mbv3_op(ef=4, m=16, n=24, k=3, s=2),
        mbv3_op(ef=3, m=24, n=24, k=3, s=1),
        mbv3_op_se(ef=3, m=24, n=40, k=5, s=2),
        mbv3_op_se(ef=3, m=40, n=40, k=5, s=1),
        mbv3_op_se(ef=3, m=40, n=40, k=5, s=1),
        mbv3_op(ef=6, m=40, n=80, k=3, s=2, act=HardSwish()),
        mbv3_op(ef=2.5, m=80, n=80, k=3, s=1, act=HardSwish()),
        mbv3_op(ef=184/80., m=80, n=80, k=3, s=1, act=HardSwish()),
        mbv3_op(ef=184/80., m=80, n=80, k=3, s=1, act=HardSwish()),
        mbv3_op_se(ef=6, m=80, n=112, k=3, s=1, act=HardSwish()),
        mbv3_op_se(ef=6, m=112, n=112, k=3, s=1, act=HardSwish()),
        mbv3_op_se(ef=6, m=112, n=160, k=5, s=2, act=HardSwish()),
        mbv3_op_se(ef=6, m=160, n=160, k=5, s=1, act=HardSwish()),
        mbv3_op_se(ef=6, m=160, n=160, k=5, s=1, act=HardSwish()),
        lib.op(Conv2d, stride=1, kernel_size=(1, 1), in_channels=160,
               out_channels=960, activation_fn=HardSwish()),
        lib.op(AdaptiveAvgPool2d, output_size=(1, 1)),
        lib.op(Conv2d, stride=1, kernel_size=(1, 1), in_channels=960,
               out_channels=1280, batch_norm=False, bias=True,
               activation_fn=HardSwish())
    ]))
        

V3_LARGE_MINIMALISTIC = dict(
    spec=([
        # stage 1
        lib.op(Conv2d, stride=2, in_channels=3, out_channels=16,
               kernel_size=(3, 3)),
        mbv3_op(ef=1, m=16, n=16, k=3),
        mbv3_op(ef=4, m=16, n=24, k=3, s=2),
        mbv3_op(ef=3, m=24, n=24, k=3, s=1),
        mbv3_op(ef=3, m=24, n=40, k=3, s=2),
        mbv3_op(ef=3, m=40, n=40, k=3, s=1),
        mbv3_op(ef=3, m=40, n=40, k=3, s=1),
        mbv3_op(ef=6, m=40, n=80, k=3, s=2),
        mbv3_op(ef=2.5, m=80, n=80, k=3, s=1),
        mbv3_op(ef=184/80., m=80, n=80, k=3, s=1),
        mbv3_op(ef=184/80., m=80, n=80, k=3, s=1),
        mbv3_op(ef=6, m=80, n=112, k=3, s=1),
        mbv3_op(ef=6, m=112, n=112, k=3, s=1),
        mbv3_op(ef=6, m=112, n=160, k=3, s=2),
        mbv3_op(ef=6, m=160, n=160, k=3, s=1),
        mbv3_op(ef=6, m=160, n=160, k=3, s=1),
        lib.op(Conv2d, stride=1, kernel_size=(1, 1), in_channels=160,
               out_channels=960),
        lib.op(AdaptiveAvgPool2d, output_size=(1, 1)),
        lib.op(Conv2d, stride=1, kernel_size=(1, 1), in_channels=960,
               out_channels=1280, batch_norm=False, bias=True)
    ]))
        

V3_SMALL = dict(
    spec=([
        # stage 1
        lib.op(Conv2d, stride=2, in_channels=3, out_channels=16,
               kernel_size=(3, 3), activation_fn=HardSwish()),
        mbv3_op(ef=1, m=16, n=16, k=3, s=2),
        mbv3_op(ef=72./16, m=16, n=24, k=3, s=2),
        mbv3_op(ef=88./24, m=24, n=24, k=3, s=1),
        mbv3_op_se(ef=4, m=24, n=40, k=5, s=2, act=HardSwish()),
        mbv3_op_se(ef=6, m=40, n=40, k=5, s=1, act=HardSwish()),
        mbv3_op_se(ef=6, m=40, n=40, k=5, s=1, act=HardSwish()),
        mbv3_op_se(ef=3, m=40, n=48, k=5, s=1, act=HardSwish()),
        mbv3_op_se(ef=3, m=48, n=48, k=5, s=1, act=HardSwish()),
        mbv3_op_se(ef=6, m=48, n=96, k=5, s=2, act=HardSwish()),
        mbv3_op_se(ef=6, m=96, n=96, k=5, s=1, act=HardSwish()),
        mbv3_op_se(ef=6, m=96, n=96, k=5, s=1, act=HardSwish()),
        lib.op(Conv2d, stride=1, kernel_size=(1, 1), in_channels=96,
               out_channels=576, activation_fn=HardSwish()),
        lib.op(AdaptiveAvgPool2d, output_size=(1, 1)),
        lib.op(Conv2d, stride=1, kernel_size=(1, 1), in_channels=576,
               out_channels=1024, batch_norm=False, bias=True,
               activation_fn=HardSwish())
    ]))
        

V3_SMALL_MINIMALISTIC = dict(
    spec=([
        # stage 1
        lib.op(Conv2d, stride=2, in_channels=3, out_channels=16,
               kernel_size=(3, 3)),
        mbv3_op(ef=1, m=16, n=16, k=3, s=2),
        mbv3_op(ef=72./16, m=16, n=24, k=3, s=2),
        mbv3_op(ef=88./24, m=24, n=24, k=3, s=1),
        mbv3_op(ef=4, m=24, n=40, k=3, s=2),
        mbv3_op(ef=6, m=40, n=40, k=3, s=1),
        mbv3_op(ef=6, m=40, n=40, k=3, s=1),
        mbv3_op(ef=3, m=40, n=48, k=3, s=1),
        mbv3_op(ef=3, m=48, n=48, k=3, s=1),
        mbv3_op(ef=6, m=48, n=96, k=3, s=2),
        mbv3_op(ef=6, m=96, n=96, k=3, s=1),
        mbv3_op(ef=6, m=96, n=96, k=3, s=1),
        lib.op(Conv2d, stride=1, kernel_size=(1, 1), in_channels=96,
               out_channels=576),
        lib.op(AdaptiveAvgPool2d, output_size=(1, 1)),
        lib.op(Conv2d, stride=1, kernel_size=(1, 1), in_channels=576,
               out_channels=1024, batch_norm=False, bias=True)
    ]))


# EdgeTPU friendly variant of MobilenetV3 that uses fused convolutions
# instead of depthwise in the early layers.
V3_EDGETPU = dict(
    spec=[
        lib.op(Conv2d, stride=2, in_channels=3, out_channels=32, 
               kernel_size=(3, 3), padding=1),
        mbv3_fused(k=3, s=1, ef=1, m=32, n=16),
        mbv3_fused(k=3, s=2, ef=8, m=16, n=32),
        mbv3_fused(k=3, s=1, ef=4, m=32, n=32),
        mbv3_fused(k=3, s=1, ef=4, m=32, n=32),
        mbv3_fused(k=3, s=1, ef=4, m=32, n=32),
        mbv3_fused(k=3, s=2, ef=8, m=32, n=48),
        mbv3_fused(k=3, s=1, ef=4, m=48, n=48),
        mbv3_fused(k=3, s=1, ef=4, m=48, n=48),
        mbv3_fused(k=3, s=1, ef=4, m=48, n=48),
        mbv3_op(k=3, s=2, ef=8, m=48, n=96),
        mbv3_op(k=3, s=1, ef=4, m=96, n=96),
        mbv3_op(k=3, s=1, ef=4, m=96, n=96),
        mbv3_op(k=3, s=1, ef=4, m=96, n=96),
        mbv3_op(k=3, s=1, ef=8, m=96, n=96, residual=False),
        mbv3_op(k=3, s=1, ef=4, m=96, n=96),
        mbv3_op(k=3, s=1, ef=4, m=96, n=96),
        mbv3_op(k=3, s=1, ef=4, m=96, n=96),
        mbv3_op(k=5, s=2, ef=8, m=96, n=160),
        mbv3_op(k=5, s=1, ef=4, m=160, n=160),
        mbv3_op(k=5, s=1, ef=4, m=160, n=160),
        mbv3_op(k=5, s=1, ef=4, m=160, n=160),
        mbv3_op(k=3, s=1, ef=8, m=160, n=192),
        lib.op(Conv2d, stride=1, in_channels=192, out_channels=1280, 
               kernel_size=(1, 1)),
    ])
        

def mobilenet(num_classes=1001,
              global_pool=True,
              depth_multiplier=1.0,
              output_stride=None,
              scope='MobilenetV3',
              conv_defs=None,
              finegrain_classification_mode=False,
              pretrained=False,
              checkpoint_path='./pretrained_models/mobilenet_v3_large.pth',
              **kwargs):
    """Creates mobilenet v3 network.
    
    Args:
        num_classes: Number of classes.
        global_pool: A boolean.
        multiplier: The multiplier applied to scale number of channels in each
            layer.
        output_stride: An integer that specifies the requested ratio of input 
            to output spatial resolution.
        scope: Scope of the operator
        conv_defs: Which version to create. Could be large/small or any 
            conv_def (see mobilenet_v3.py for examples).
        finegrain_classification_mode: When set to True, the model
            will keep the last layer large even for small multipliers. Following
            https://arxiv.org/abs/1801.04381
            it improves performance for ImageNet-type of problems.
              *Note* ignored if final_endpoint makes the builder exit earlier.
        **kwargs: Passed directly to mobilenet.MobileNet.
        
    Returns:
        An instance of mobilenet.MobileNet.
    """
    if conv_defs is None:
        conv_defs = V3_LARGE
        
    if finegrain_classification_mode:
        conv_defs = copy.deepcopy(conv_defs)
        conv_defs['spec'][-1] = conv_defs['spec'][-1]._replace(
            multiplier_func=lib.half_depth_multiplier)
        
    model = lib.MobileNet(conv_defs=conv_defs,
                          num_classes=num_classes,
                          global_pool=global_pool,
                          multiplier=depth_multiplier,
                          output_stride=output_stride,
                          scope=scope,
                          **kwargs)
    if pretrained:
        _load_state_dict(model, num_classes, checkpoint_path)
    return model
    
    
def _load_state_dict(model, num_classes, checkpoint_path):
    """Load pretrained weights."""
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        if num_classes is None or num_classes != 1001:
            num_layers = len(model._layers)
            state_dict.pop('_layers.{}.weight'.format(num_layers - 1))
            state_dict.pop('_layers.{}.bias'.format(num_layers - 1))
        model.load_state_dict(state_dict, strict=False)
        print('Load pretrained weights successfully.')
    else:
        raise ValueError('`checkpoint_path` does not exist.')
    
    
def wrapped_partial(func, new_defaults=None,
                    **kwargs):
    """Partial function with new default parameters and updated docstring."""
    if not new_defaults:
        new_defaults = {}
        
    def func_wrapper(*f_args, **f_kwargs):
        new_kwargs = dict(new_defaults)
        new_kwargs.update(f_kwargs)
        return func(*f_args, **new_kwargs)
    
    functools.update_wrapper(func_wrapper, func)
    partial_func = functools.partial(func_wrapper, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


large = wrapped_partial(mobilenet, conv_defs=V3_LARGE)
small = wrapped_partial(mobilenet, 
                        new_defaults={'model_name': 'small'},
                        conv_defs=V3_SMALL)
edge_tpu = wrapped_partial(mobilenet,
                           new_defaults={'scope': 'MobilenetEdgeTPU',
                                         'model_name': 'edgeTPU'},
                           conv_defs=V3_EDGETPU)


# Minimalistic model that does not have Squeeze Excite blocks,
# Hardswish, or 5x5 depthwise convolution.
# This makes the model very friendly for a wide range of hardware
large_minimalistic = wrapped_partial(mobilenet,
                                     new_defaults={'model_name': 
                                         'large_minimalistic'},
                                     conv_defs=V3_LARGE_MINIMALISTIC)
small_minimalistic = wrapped_partial(mobilenet, 
                                     new_defaults={'model_name': 
                                         'small_minimalistic'},
                                     conv_defs=V3_SMALL_MINIMALISTIC) 