from numpy import sqrt


def initialise_zero_bias(conv_layer):
    '''
    Initialises a bias of convolutional layer with zeros.
    See https://arxiv.org/pdf/1502.01852.pdf for detail 
    Parameters:
        - conv_layer -- a 2D convolution layer, torch.nn.Conv2d
    Returns:
        - None
    '''
    if conv_layer.bias is not None:
        conv_layer.bias.data.fill_(0.0)


def initialise_linear_layer(linear_layer, std_scale_factor=3.0):
    '''
    Initialises a linear layer (weights and bias).
    See https://arxiv.org/pdf/1502.01852.pdf for detail 
    Parameters:
        - linear_layer -- layer for a linear transformation to the incoming data, torch.nn.Linear
        - std_scale_factor -- scales the std by sqrt(std_scale_factor), float. Defaults to 3.0
    Returns:
        - None
    '''
    (_, N) = linear_layer.weight.size()
    K = sqrt(std_scale_factor * (2.0 / N))
    linear_layer.weight.data.uniform_(-K, K)
    initialise_zero_bias(linear_layer)


def initialise_conv_layer(conv_layer, std_scale_factor=3.0):
    '''
    Initialises a convolutional layer (weights and bias).
    See https://arxiv.org/pdf/1502.01852.pdf for detail 
    Parameters:
        - conv_layer -- a 2D convolution layer, torch.nn.Conv2d
        - std_scale_factor -- scales the std by sqrt(std_scale_factor), float. Defaults to 3.0
    Returns:
        - None
    '''
    (_, n_in, height, width) = conv_layer.weight.size()
    N = n_in * height * width
    K = sqrt(std_scale_factor * (2.0 / N))
    conv_layer.weight.data.uniform_(-K, K)
    initialise_zero_bias(conv_layer)
