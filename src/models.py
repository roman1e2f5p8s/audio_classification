import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_blocks import VGGishConvBlock
from .init_layers import initialise_linear_layer


class VGGish(nn.Module):
    '''
    Class of the VGGish model
    See https://www.researchgate.net/publication/
    330925933_Simple_CNN_and_vggish_model_for_high-level_sound_categorization_
    within_the_Making_Sense_of_Sounds_challenge
    '''
    def __init__(self, params):
        '''
        Initialisation
        Arguments:
            - params -- model parameters, hparams.HParamsFromYAML
        Attributes:
            - classes_number -- number of classes in a multiclass classfication problem, int > 0
            - conv_blocks_number -- number of the convolutional blocks, int > 0. Defaults to 4
            - kernel_size -- size of the convolving kernel, tuple like (a, b), where a, b are int > 0.
                Defaults to (3, 3)
            - stride -- stride of the convolution, tuple like (a, b), where a, b are int > 0.
                Defaults to (1, 1)
            - padding -- zero-padding added to both sides of the input, tuple like (a, b),
                where a, b are int > 0. Defaults to (1, 1)
            - add_bias_to_blocks -- whether add a learnable bias to the output of convolutional blocks,
                bool. Defaults to False
            - init_layers_manually -- whether initialise the layers using the method descried in 
                https://arxiv.org/pdf/1502.01852.pdf , bool. Defaults to False 
            - std_scale_factor -- if init_layer_manually is True, scales the std by 
                sqrt(std_scale_factor), float. Defaults to 3.0
            - max_pool_kernel_size -- the size of the window to take a max over, tuple like (a, b),
                where a, b are int > 0. Defaults to (2, 2)
            - max_pool_stride – the stride of the window, tuple like (a, b), where a, b are int > 0.
                Defaults to (2, 2)
            - add_bias_to_fc -- whether add an additive bias to fully connected layer,
                bool. Defaults to True
        '''
        super(VGGish, self).__init__()

        self.classes_number = params.classes_number
        self.conv_blocks_number = params.conv_blocks_number
        self.kernel_size = (params.kernel_size_1st_dim, params.kernel_size_2nd_dim)
        self.stride = (params.stride_1st_dim, params.stride_2nd_dim)
        self.padding = (params.padding_1st_dim, params.padding_2nd_dim)
        self.add_bias_to_blocks = params.add_bias_to_blocks
        self.init_layers_manually = params.init_layers_manually
        self.std_scale_factor = params.std_scale_factor
        self.max_pool_kernel_size = (params.max_pool_kernel_size_1st_dim,
                params.max_pool_kernel_size_2nd_dim)
        self.max_pool_stride = (params.max_pool_stride_1st_dim, params.max_pool_stride_2nd_dim)
        self.add_bias_to_fc = params.add_bias_to_fc

        # add convolutional blocks
        n_out_channels = None
        for k in range(1, self.conv_blocks_number + 1):
            k_ = -4 if k == 1 else k
            n_out_channels = 2**(6 + k - 1)
            setattr(self, 'conv_block_{}'.format(k),
                    VGGishConvBlock(
                        in_channels_number=2**(6 + k_ - 2),
                        out_channels_number=n_out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        add_bias=self.add_bias_to_blocks,
                        init_layers_manually=self.init_layers_manually,
                        std_scale_factor=self.std_scale_factor,
                        max_pool_kernel_size=self.max_pool_kernel_size,
                        max_pool_stride=self.max_pool_stride))

        # add the final fully connected layer
        self.fc_layer = nn.Linear(
                in_features=n_out_channels,
                out_features=self.classes_number,
                bias=self.add_bias_to_fc)

        # initialise the fully connected layer at the end as described in
        # https://arxiv.org/pdf/1502.01852.pdf 
        if self.init_layers_manually:
            initialise_linear_layer(self.fc_layer, std_scale_factor=self.std_scale_factor)

    def forward(self, input_):
        '''
        Overrides nn.Module.forward method.
        Defines the computation performed at every call
        Parameters:
            - input_ -- input
        Returns:
            - out -- output
        '''
        (batch_size, input_frames_number, mels_number) = input_.shape
        # reshape input_ to 4D tensor of shape (batch_size, 1, input_frames_number, mels_number)
        x = input_.view(-1, 1 , input_frames_number, mels_number)

        for k in range(1, self.conv_blocks_number + 1):
            x = self.__getattr__('conv_block_{}'.format(k))(x)

        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])
        out = F.log_softmax(self.fc_layer(x), dim=-1)
        return out
