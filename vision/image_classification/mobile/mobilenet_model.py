"""MobileNet model for Keras.

Related papers
- https://arxiv.org/abs/1704.04861

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import layers


def _conv_block(input_tensor,
                filters,
                alpha,
                kernel=(3, 3),
                strides=(1, 1)):
    """Adds an initial convolution layer (with batch normalization and relu).

    Arguments:
      input_tensor: input tensor
      filters: Integer, the dimensionality of the output space.
      alpha: controls the width of the network. - If `alpha` < 1.0,
        proportionally decreases the number of filters in each layer. - If
        `alpha` > 1.0, proportionally increases the number of filters in each
        layer. - If `alpha` = 1, default number of filters from the paper are
        used at each layer.
      kernel: An integer or tuple/list of 2 integers, specifying the width and
        height of the 2D convolution window. Can be a single integer to
        specify the same value for all spatial dimensions.
      strides: An integer or tuple/list of 2 integers, specifying the strides
        of the convolution along the width and height. Can be a single integer
        to specify the same value for all spatial dimensions.

    Returns:
      Output tensor of block.
    """

    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:  # channels_last
        channel_axis = -1

    filters = int(filters * alpha)
    x = layers.ZeroPadding2D(
        padding=((0, 1), (0, 1)),
        name='conv1_pad')(input_tensor)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel,
        padding='valid',
        use_bias=False,
        strides=strides,
        name='conv1')(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)

    return x


def _depthwise_conv_block(input_tensor,
                          pointwise_conv_filters,
                          alpha,
                          depth_multiplier=1,
                          strides=(1, 1),
                          block_id=1):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu, pointwise convolution,
    batch normalization and relu activation.

    Arguments:
      input_tensor: input tensor.
      pointwise_conv_filters: Integer, the dimensionality of the output space.
      alpha: controls the width of the network. - If `alpha` < 1.0,
        proportionally decreases the number of filters in each layer. - If
        `alpha` > 1.0, proportionally increases the number of filters in each
        layer. - If `alpha` = 1, default number of filters from the paper are
        used at each layer.
      depth_multiplier: The number of depthwise convolution output channels
        for each input channel. The total number of depthwise convolution
        output channels will be equal to `filters_in * depth_multiplier`.
      strides: An integer or tuple/list of 2 integers, specifying the strides
        of the convolution along the width and height. Can be a single integer
        to specify the same value for all spatial dimensions.
      block_id: Integer, a unique identification designating the block number.

    Returns:
      Output tensor of block.
    """

    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:  # channels_last
        channel_axis = -1

    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = input_tensor
    else:
        x = layers.ZeroPadding2D(
            padding=((0, 1), (0, 1)),
            name='conv_pad_%d' % block_id)(input_tensor)

    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        padding='same' if strides == (1, 1) else 'valid',
        depth_multiplier=depth_multiplier,
        strides=strides,
        use_bias=False,
        name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        name='conv_dw_%d_bn' % block_id)(x)
    x = layers.Activation('relu', name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(
        pointwise_conv_filters, (1, 1),
        padding='same',
        use_bias=False,
        strides=(1, 1),
        name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        name='conv_pw_%d_bn' % block_id)(x)
    x = layers.Activation('relu', name='conv_pw_%d_relu' % block_id)(x)

    return x


def mobilenet(num_classes=1000,
              batch_size=None,
              resolution_scale=224,
              width_multiplier=1.0,
              depth_multiplier=1,
              dropout=1e-3):
    """Instantiates the architecture.

    Arguments:
      width_multiplier: Controls the width of the network. This is known as the width
        multiplier in the MobileNet paper. - If `alpha` < 1.0, proportionally
        decreases the number of filters in each layer. - If `alpha` > 1.0,
        proportionally increases the number of filters in each layer. - If
        `alpha` = 1, default number of filters from the paper are used at each
        layer. Default to 1.0.
      resolution_scale: 128, 160, 192, 224
      depth_multiplier: Depth multiplier for depthwise convolution. Default to 1.0.
      dropout: Dropout rate. Default to 0.001.
      num_classes: `int` number of classes for image classification.
      batch_size: Size of the batches for each step.

    Returns:
      A Keras model instance.

    """

    input_shape = (resolution_scale, resolution_scale, 3)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    x = img_input

    if backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        shape = (int(1024 * width_multiplier), 1, 1)
    else:  # channels_last
        shape = (1, 1, int(1024 * width_multiplier))

    x = _conv_block(x, 32, width_multiplier, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, width_multiplier, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 128, width_multiplier, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, width_multiplier, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 256, width_multiplier, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, width_multiplier, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 512, width_multiplier, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, width_multiplier, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, width_multiplier, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, width_multiplier, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, width_multiplier, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, width_multiplier, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 1024, width_multiplier, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, width_multiplier, depth_multiplier, block_id=13)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape(shape, name='reshape')(x)
    x = layers.Dropout(dropout, name='dropout')(x)

    x = layers.Conv2D(num_classes, (1, 1), padding='same', name='conv_preds')(x)
    x = layers.Reshape((num_classes,), name='reshape_')(x)

    x = layers.Activation(activation='softmax', name='predictions', dtype='float32')(x)

    # Create model.
    return models.Model(img_input, x, name='mobilenet_%0.2f_%d' % (width_multiplier, resolution_scale))

