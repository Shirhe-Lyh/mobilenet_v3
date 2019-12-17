# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 19:22:55 2019

@author: shirhe-lyh


Reference:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/
        mobilenet/conv_blocks.py
"""

import functools
import torch


_BATCH_NORM_PARAMS = {
    'eps': 0.001,
    'momentum': 0.997,
    'affine': True,
}


def _same_padding(x, kernel_size=3, dilation=1):
    """The padding method used in slim.conv2d(..., padding='SAME').
    
    Note: only for stride = 2 or stride = 3.
    """
    k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    d = dilation if isinstance(dilation, int) else dilation[0]
    pad_total = d * (k - 1) - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    x_padded = torch.nn.functional.pad(
        x, pad=(pad_beg, pad_end, pad_beg, pad_end))
    return x_padded


def _fixed_padding(inputs, kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.
    
    Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        kernel_size: The kernel to be used in the conv2d or max_pool2d 
            operation. 
        rate: An integer, rate for atrous convolution.
        
    Returns:
        padded_inputs: A tensor of size [batch, height_out, width_out, 
            channels] with the input, either intact (if kernel_size == 1) or 
            padded (if kernel_size > 1).
    """
    if not isinstance(kernel_size, int):
        kernel_size = kernel_size[0]
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = torch.nn.functional.pad(
        inputs, pad=(pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _split_divisible(num, num_ways, divisible_by=8):
    """Evenly splits num, num_ways so each piece is a multiple of divisible by."""
    assert num % divisible_by == 0
    assert num / num_ways >= divisible_by
    # Note: want to round down, we adjust each split to match the total.
    base = num // num_ways // divisible_by * divisible_by
    result = []
    accumulated = 0
    for i in range(num_ways):
        r = base
        while accumulated + r < num * (i + 1) / num_ways:
            r += divisible_by
        result.append(r)
        accumulated += r
    assert accumulated == num
    return result


def expand_input_by_factor(n, divisible_by=8):
    return lambda num_inputs, **_: _make_divisible(num_inputs * n, divisible_by)


class SplitConv(torch.nn.Module):
    """Creates a split convolution."""
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_ways,
                 divisible_by=8,
                 batch_norm=False,
                 activation_fn=torch.nn.ReLU(inplace=True)):
        """Constructor.
        
        Args:
            in_channels: Number of input filters.
            out_channels: Number of output filters.
            num_ways: Number of blocks to split by.
            divisible_by: Make sure that every part is divisible by this.
        """
        super(SplitConv, self).__init__()
        
        if num_ways == 1 or min(in_channels // num_ways,
                                out_channels // num_ways) < divisible_by:
            input_splits = [in_channels]
            output_splits = [out_channels]
        else:
            input_splits = _split_divisible(in_channels, num_ways, 
                                            divisible_by=divisible_by)
            output_splits = _split_divisible(out_channels, num_ways,
                                             divisible_by=divisible_by)
            
        branches = []
        for in_size, out_size in zip(input_splits, output_splits):
            branch = [torch.nn.Conv2d(in_channels=in_size, 
                                      out_channels=out_size, 
                                      kernel_size=1,
                                      bias=False)]
            if batch_norm:
                branch += [torch.nn.BatchNorm2d(num_features=out_size,
                                                **_BATCH_NORM_PARAMS)]
            if activation_fn is not None:
                branch += [activation_fn]
            branches.append(torch.nn.Sequential(*branch))
        self._input_splits = input_splits
        self._branches = torch.nn.Sequential(*branches)
        
    def forward(self, input_tensor):
        inputs = [input_tensor]
        if len(self._input_splits) > 1:
            inputs = torch.split(input_tensor, self._input_splits, dim=1)
        outputs = [branch(x) for branch, x in zip(self._branches, inputs)]
        return torch.cat(outputs, dim=1)
    
    
class DepthwiseConv(torch.nn.Module):
    """Depthwise convolution."""
    
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding='SAME',
                 batch_norm=False,
                 activation_fn=torch.nn.ReLU(inplace=True)):
        """Constructor.
        
        Args:
            in_channels: The number of input filters.
            kernel_size: A list of length 2: [kernel_height, kernel_width] of 
                the filters. Can be an int if both values are the same.
            stride: A list of length 2: [stride_height, stride_width], 
                specifying the depthwise convolution stride. Can be an int if 
                both strides are the same.
            padding: One of 'VALID' or 'SAME'.
            dilation: A list of length 2: [rate_height, rate_width], 
                specifying the dilation rates for atrous convolution. 
                Can be an int if both rates are the same. If any value is 
                larger than one, then both stride values need to be one.
            batch_norm: If True, normalization function to use instead of 
                `biases`. Default set to False for no normalizer function.
            activation_fn: Activation function. The default value is a ReLU 
                function. Explicitly set it to None to skip it and maintain 
                a linear activation.
        """
        super(DepthwiseConv, self).__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        else:
            kernel_size = tuple(kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        else:
            stride = tuple(stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        else:
            dilation = tuple(dilation)
            
        self._without_padding = (padding == 'SAME' and stride[0] == 1)
        if self._without_padding:
            padding_h = (kernel_size[0] - 1) * dilation[0] // 2
            padding_w = (kernel_size[1] - 1) * dilation[1] // 2
            padding = (padding_h, padding_w)
        else:
            padding = (0, 0)
            self._kernel_size = kernel_size
            self._dilation = dilation
            
        layers = [torch.nn.Conv2d(in_channels=in_channels,
                                  out_channels=in_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=in_channels,
                                  bias=False)]
        if batch_norm:
            layers += [torch.nn.BatchNorm2d(num_features=in_channels,
                                            **_BATCH_NORM_PARAMS)]
        if activation_fn is not None:
            layers += [activation_fn]
        self._layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        if not self._without_padding:
            x = _same_padding(x, kernel_size=self._kernel_size,
                              dilation=self._dilation)
        x = self._layers(x)
        return x
    
    
class ExpandedConv(torch.nn.Module):
    """Depthwise Convolution Block with expansion.
    
    Builds a composite convolution that has the following structure
    expansion (1x1) -> depthwise (kernel_size) -> projection (1x1)
    """
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion_size=expand_input_by_factor(6),
                 stride=1,
                 dilation=1,
                 kernel_size=(3, 3),
                 residual=True,
                 batch_norm=True,
                 split_projection=1,
                 split_expansion=1,
                 split_divisible_by=8,
                 expansion_transform=None,
                 depthwise_location='expansion',
                 use_explicit_padding=False,
                 padding='SAME',
                 inner_activation_fn=torch.nn.ReLU(inplace=True),
                 depthwise_activation_fn=None,
                 projection_activation_fn=torch.nn.Identity(),
                 depthwise_fn=DepthwiseConv,
                 expansion_fn=SplitConv,
                 projection_fn=SplitConv):
        """Constructor.
        
        Args:
            in_channels: Number of input filters.
            out_channels: Number of output filters.
            expansion_size: The size of expansion, could be a constant or a 
                callable. If latter it will be provided 'num_inputs' as an 
                input. For forward compatibility it should accept arbitrary 
                keyword arguments. Default will expand the input by factor 
                of 6.
            stride: Depthwise stride.
            dilation: Depthwise rate.
            kernel_size: depthwise kernel.
            residual: Whether to include residual connection between input
                and output.
            batch_norm: Batchnorm or otherwise.
            split_projection: How many ways to split projection operator
                (that is conv expansion->bottleneck)
            split_expansion: How many ways to split expansion op
                (that is conv bottleneck->expansion) ops will keep depth 
                divisible by this value.
            split_divisible_by: Make sure every split group is divisible by 
                this number.
            expansion_transform: Optional function that takes expansion
                as a single input and returns output.
            depthwise_location: Where to put depthwise covnvolutions supported
                values None, 'input', 'output', 'expansion'.
            use_explicit_padding: Use 'VALID' padding for convolutions, but 
                prepad inputs so that the output dimensions are the same as 
                if 'SAME' padding were used.
            padding: Padding type to use if `use_explicit_padding` is not set.
            inner_activation_fn: Activation function to use in all inner 
                convolutions.
            depthwise_activation_fn: Activation function to use for deptwhise
                only. If both inner_activation_fn and depthwise_activation_fn 
                are provided, depthwise_activation_fn takes precedence over 
                inner_activation_fn.
            project_activation_fn: Activation function for the project layer.
                (note this layer is not affected by inner_activation_fn)
            depthwise_fn: Depthwise convolution function.
            expansion_fn: Expansion convolution function. If use custom 
                function then "split_expansion" and "split_divisible_by" will 
                be ignored.
            projection_fn: Projection convolution function. If use custom 
                function then "split_projection" and "split_divisible_by" will 
                be ignored.
                
        Raises:
            ValueError: If "depthwise_location" not in [None, 'input', 'output',
                'expansion'] or padding != 'SAME' when use explicit padding.
        """
        super(ExpandedConv, self).__init__()
        
        if depthwise_location not in [None, 'input', 'output', 'expansion']:
            raise ValueError('%r is unknown value for depthwise_location' %
                             depthwise_location)
        if use_explicit_padding:
            if padding != 'SAME':
                raise ValueError('`use_explicit_padding` should only be used '
                                 'with "SAME" padding.')
            padding = 'VALID'
        
        if callable(expansion_size):
            inner_size = expansion_size(num_inputs=in_channels)
        else:
            inner_size = expansion_size
            
        # Expansion conv
        if inner_size > in_channels:
            if expansion_fn == SplitConv:
                expansion_fn = functools.partial(
                    expansion_fn,
                    num_ways=split_expansion,
                    divisible_by=split_divisible_by)
            expansion_fn = functools.partial(
                expansion_fn,
                in_channels=in_channels,
                out_channels=inner_size,
                batch_norm=batch_norm,
                activation_fn=inner_activation_fn)
            self._expansion_fn = expansion_fn()
            
        # Depthwise conv
        if depthwise_activation_fn is None:
            depthwise_activation_fn = inner_activation_fn
        if depthwise_location == 'input':
            depthwise_channels = in_channels
            depthwise_act_fn = None
        elif depthwise_location == 'output':
            depthwise_channels = out_channels
            depthwise_act_fn = None
        elif depthwise_location == 'expansion':
            depthwise_channels = max(in_channels, inner_size)
            depthwise_act_fn = depthwise_activation_fn
        else:
            depthwise_channels = 1
            depthwise_act_fn = depthwise_activation_fn
        depthwise_func = functools.partial(
            depthwise_fn,
            in_channels=depthwise_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            batch_norm=batch_norm,
            activation_fn=depthwise_act_fn)
        self._depthwise_func = depthwise_func()
        
        # Projection conv
        if projection_fn == SplitConv:
            projection_fn = functools.partial(
                projection_fn,
                num_ways=split_expansion,
                divisible_by=split_divisible_by)
        projection_fn = functools.partial(
            projection_fn,
            in_channels=max(in_channels, inner_size),
            out_channels=out_channels,
            batch_norm=batch_norm,
            activation_fn=projection_activation_fn)
        self._projection_fn = projection_fn()
        
        self._expansion_transform = None
        if expansion_transform is not None:
            expansion_transform = functools.partial(
                expansion_transform,
                in_channels=max(in_channels, inner_size),
                out_channels=max(in_channels, inner_size))
            self._expansion_transform = expansion_transform()
        
        self._in_channels = in_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._dilation = dilation
            
        self._residual = residual
        self._inner_size = inner_size
        self._depthwise_location = depthwise_location
        self._use_explicit_padding = use_explicit_padding
        
    def forward(self, x, endpoints=None):
        """Forward computation.
        
        Args:
            x: A float32 tensor with shape [batch, height, width, depth].
            endpoints: An optional dictionary into which intermediate endpoints
                are placed. The keys "expansion_output", "depthwise_output",
                "projection_output" and "expansion_transform" are always 
                populated, even if the corresponding functions are not invoked.
                
        Returns:
            The computed outputs.
        """
        net = x
        
        if self._depthwise_location == 'input':
            if self._use_explicit_padding:
                net = _fixed_padding(net, self._kernel_size, self._dilation)
            net = self._depthwise_func(net)
            if endpoints is not None:
                endpoints['depthwise_output'] = net
                
        if self._inner_size > self._in_channels:
            net = self._expansion_fn(net)
            if endpoints is not None:
                endpoints['expansion_output'] = net
                
        if self._depthwise_location == 'expansion':
            if self._use_explicit_padding:
                net = _fixed_padding(net, self._kernel_size, self._dilation)
            net = self._depthwise_func(net)
            if endpoints is not None:
                endpoints['depthwise_output'] = net
                
        if self._expansion_transform is not None:
            net = self._expansion_transform(net)
            
        # Note in contrast with expansion, we always have
        # projection to produce the desired output size.
        net = self._projection_fn(net)
        if endpoints is not None:
            endpoints['projection_output'] = net
            
        if self._depthwise_location == 'output':
            if self._use_explicit_padding:
                net = _fixed_padding(net, self._kernel_size, self._dilation)
            net = self._depthwise_func(net)
            if endpoints is not None:
                endpoints['depthwise_output'] = net
        
        if callable(self._residual):  # Custom residual
            net = self._residual(input_tensor=x, output_tensor=net)
        elif (self._residual and
              # stride check enforces that we don't add residuals when spatial
              # dimensions are None
              self._stride == 1 and
              # Depth matches
              net.shape[1] == x.shape[1]):
            net += x
        return net


class SqueezeExcitation(torch.nn.Module):
    """Squeeze excite bolck for MobileNet V3."""
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 divisible_by=8,
                 squeeze_factor=3,
                 inner_activation_fn=torch.nn.ReLU(inplace=True),
                 gating_fn=torch.nn.Sigmoid()):
        """Constructor.
        
        Args:
            in_channels: Number of input filters.
            out_channels: Number of output filters.
            divisible_by: Ensures all inner dimensions are divisible by this
                Number.
            squeeze_factor: The factor of squeezing in the inner fully 
                connected layer.
            inner_activation_fn: Non-linearity to be used in inner layer.
            gating_fn: Non-linearity to be used for final gating-function.
        """
        super(SqueezeExcitation, self).__init__()
        
        squeeze_channels = _make_divisible(in_channels / squeeze_factor,
                                           divisor=divisible_by)
        
        layers = [torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                  torch.nn.Conv2d(in_channels=in_channels,
                                  out_channels=squeeze_channels,
                                  kernel_size=1,
                                  bias=True),
                  inner_activation_fn,
                  torch.nn.Conv2d(in_channels=squeeze_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  bias=True),
                  gating_fn]
        self._layers = torch.nn.Sequential(*layers)
        
    def forward(self, input_tensor, squeeze_input_tensor=None):
        """Forward computation.
        
        Args:
            input_tensor: A tensor with shape [batch, height, width, depth].
            squeeze_input_tensor: A custom tensor to use for computing gating
                activation. If provided the result will be input_tensor * SE(
                squeeze_input_tensor) instead of input_tensor * SE(
                input_tensor).
                
        Returns:
            Gated input tensor. (e.g. X * SE(X))
        """
        if squeeze_input_tensor is None:
            squeeze_input_tensor = input_tensor
        squeeze_excite = self._layers(squeeze_input_tensor)
        result = input_tensor * squeeze_excite
        return result