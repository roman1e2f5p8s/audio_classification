import torch
import torch.nn as nn
import torch.nn.functional as F

from .init_layers import initialise_conv_layer


class VGGishConvBlock(nn.Module):
    '''
    Class for creating VGGish convolutional blocks
    See https://www.researchgate.net/publication/
    330925933_Simple_CNN_and_vggish_model_for_high-level_sound_categorization_
    within_the_Making_Sense_of_Sounds_challenge
    '''
    def __init__(self, in_channels_number, out_channels_number, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1), add_bias=False, init_layers_manually=True, std_scale_factor=3.0,
            max_pool_kernel_size=(2, 2), max_pool_stride=(2, 2)):
        '''
        Initialisation
        Arguments:
            - in_channels_number -- number of channels in the input audio, int > 0
            - out_channels_number -- number of channels produced by the convolution, int > 0
            - kernel_size -- size of the convolving kernel, tuple like (a, b), where a, b are int > 0.
                Defaults to (3, 3)
            - stride -- stride of the convolution, tuple like (a, b), where a, b are int > 0.
                Defaults to (1, 1)
            - padding -- zero-padding added to both sides of the input, tuple like (a, b),
                where a, b are int > 0. Defaults to (1, 1)
            - add_bias -- whether add a learnable bias to the output, bool. Defaults to False
            - init_layers_manually -- whether initialise the layers using the method descried in 
                https://arxiv.org/pdf/1502.01852.pdf , bool. Defaults to True 
            - std_scale_factor -- if init_layer_manually is True, scales the std by 
                sqrt(std_scale_factor), float. Defaults to 3.0
            - max_pool_kernel_size -- the size of the window to take a max over, tuple like (a, b),
                where a, b are int > 0. Defaults to (2, 2)
            - max_pool_stride – the stride of the window, tuple like (a, b), where a, b are int > 0.
                Defaults to (2, 2)
        '''
        super(VGGishConvBlock, self).__init__()

        self.in_channels_number = in_channels_number
        self.out_channels_number = out_channels_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.add_bias = add_bias
        self.init_layers_manually = init_layers_manually
        self.std_scale_factor = std_scale_factor
        self.max_pool_kernel_size = max_pool_kernel_size
        self.max_pool_stride = max_pool_stride
        
        # apply the first 2D convolution layer over an input signal
        self.conv_1 = nn.Conv2d(
                in_channels=in_channels_number,
                out_channels=out_channels_number,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=add_bias)

        # apply the second 2D convolution layer over an input signal
        self.conv_2 = nn.Conv2d(
                in_channels=out_channels_number,
                out_channels=out_channels_number,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=add_bias)

        # apply the batch normalisation layers
        self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels_number)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels_number)

        # initialise the layers as described in https://arxiv.org/pdf/1502.01852.pdf 
        if init_layers_manually:
            initialise_conv_layer(self.conv_1, std_scale_factor=std_scale_factor)
            initialise_conv_layer(self.conv_2, std_scale_factor=std_scale_factor)

    def forward(self, input_):
        '''
        Overrides nn.Module.forward method.
        Defines the computation performed at every call
        Parameters:
            - input_ -- input
        Returns:
            - out -- output
        '''
        x = F.relu(self.batch_norm_1(self.conv_1(input_)))
        x = F.relu(self.batch_norm_2(self.conv_2(x)))
        out = F.max_pool2d(x, kernel_size=self.max_pool_kernel_size, stride=self.max_pool_stride)
        return out
