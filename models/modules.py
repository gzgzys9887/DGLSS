from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiSparseTensor import _get_coordinate_map_key
from MinkowskiEngine import SparseTensor
from MinkowskiEngine.MinkowskiTensorField import TensorField

import re
from collections import OrderedDict

import pdb

def get_child_dict(params, key=None):
        """
        Constructs parameter dictionary for a network module.
        Args:
            params (dict): a parent dictionary of named parameters.
            key (str, optional): a key that specifies the root of the child dictionary.
        Returns:
            child_dict (dict): a child dictionary of model parameters.
        """
        if params is None:
            return None
        if key is None or (isinstance(key, str) and key == ''):
            return params

        key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
        if not any(filter(key_re.match, params.keys())):  # handles nn.DataParallel
            key_re = re.compile(r'^module\.{0}\.(.+)'.format(re.escape(key)))
        child_dict = OrderedDict(
            (key_re.sub(r'\1', k), value) for (k, value)
            in params.items() if key_re.match(k) is not None)
        return child_dict
    
class MinkowskiConvolution(ME.MinkowskiConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=-1,
        stride=1,
        dilation=1,
        bias=False,
        dimension=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            dimension=dimension,
        )
    
    def forward(self, input, coordinates=None, params=None):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension
        if params is None:
            out = super(MinkowskiConvolution, self).forward(input, coordinates)
        else:
            kernel, bias = params.get('kernel'), params.get('bias')
            if kernel is None:
                kernel = self.kernel
            if bias is None:
                bias = self.bias
            
            if self.use_mm:
                # If the kernel_size == 1, the convolution is simply a matrix
                # multiplication
                out_coordinate_map_key = input.coordinate_map_key
                outfeat = input.F.mm(kernel)
            else:
                # Get a new coordinate_map_key or extract one from the coords
                out_coordinate_map_key = _get_coordinate_map_key(
                    input, coordinates, self.kernel_generator.expand_coordinates
                )
                outfeat = self.conv.apply(
                    input.F,
                    kernel,
                    self.kernel_generator,
                    self.convolution_mode,
                    input.coordinate_map_key,
                    out_coordinate_map_key,
                    input._manager,
                )
            if bias is not None:
                outfeat += bias

            out = SparseTensor(
                outfeat,
                coordinate_map_key=out_coordinate_map_key,
                coordinate_manager=input._manager,
            )
            
        return out
    
class MinkowskiConvolutionTranspose(ME.MinkowskiConvolutionTranspose):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=-1,
        stride=1,
        dilation=1,
        bias=False,
        dimension=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            dimension=dimension,
        )
    
    def forward(self, input, coordinates=None, params=None):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension
        
        if params is None:
            out = super(MinkowskiConvolutionTranspose, self).forward(input, coordinates)
        else:
            kernel, bias = params.get('kernel'), params.get('bias')
            if kernel is None:
                kernel = self.kernel
            if bias is None:
                bias = self.bias
            
            if self.use_mm:
                # If the kernel_size == 1, the convolution is simply a matrix
                # multiplication
                out_coordinate_map_key = input.coordinate_map_key
                outfeat = input.F.mm(kernel)
            else:
                # Get a new coordinate_map_key or extract one from the coords
                out_coordinate_map_key = _get_coordinate_map_key(
                    input, coordinates, self.kernel_generator.expand_coordinates
                )
                outfeat = self.conv.apply(
                    input.F,
                    kernel,
                    self.kernel_generator,
                    self.convolution_mode,
                    input.coordinate_map_key,
                    out_coordinate_map_key,
                    input._manager,
                )
            if bias is not None:
                outfeat += bias

            out = SparseTensor(
                outfeat,
                coordinate_map_key=out_coordinate_map_key,
                coordinate_manager=input._manager,
            )
            
        return out
    
class MinkowskiBatchNorm(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(MinkowskiBatchNorm, self).__init__()
        
        self.bn = BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, input, params=None):
        output = self.bn(input.F, get_child_dict(params, 'bn'))
        if isinstance(input, TensorField):
            return TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )
            
class BatchNorm1d(nn.BatchNorm1d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
        )
    
    def forward(self, input, params=None):
        if params is None:
            return super(BatchNorm1d, self).forward(input)
        else:
            weight, bias = params.get('weight'), params.get('bias')
            if weight is None:
                weight = self.weight
            if bias is None:
                bias = self.bias
            
            self._check_input_dim(input)
            
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked is not None:  # type: ignore
                    self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            r"""
            Decide whether the mini-batch stats should be used for normalization rather than the buffers.
            Mini-batch stats are used in training mode, and in eval mode when buffers are None.
            """
            if self.training:
                bn_training = True
            else:
                bn_training = (self.running_mean is None) and (self.running_var is None)

            r"""
            Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
            passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
            used for normalization (i.e. in eval mode when buffers are not None).
            """
            assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
            assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                weight, bias, bn_training, exponential_average_factor, self.eps)

class Sequential(nn.Sequential):
  def __init__(self, *args):
    super(Sequential, self).__init__(*args)

  def forward(self, x, params=None):
    if params is None:
      for module in self:
        x = module(x, params=None)
    else:
      for name, module in self._modules.items():
        x = module(x, params=get_child_dict(params, name))
    return x

class MetricLearner(nn.Module):
    def __init__(self):
        super(MetricLearner, self).__init__()
        self.fc1 = nn.Linear(96, 96)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(96, 64)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        
        return self.fc2(self.relu(self.fc1(x)))
        