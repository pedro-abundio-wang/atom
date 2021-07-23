"""ResNet50 model for Keras.

Related papers/blogs:
- https://arxiv.org/abs/1603.05027

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import layers
from tensorflow.python.keras import backend
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers

L2_WEIGHT_DECAY = 1e-4


def bottleneck_block(input_tensor,
                     filters,
                     kernel_size=3,
                     stride=1,
                     conv_shortcut=False,
                     name=None):
    """A residual block.

    Arguments:
        input_tensor: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
          otherwise identity shortcut.
        name: string, block label.

    Returns:
      Output tensor for the residual block.
    """

    if backend.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    preact = layers.BatchNormalization(
        axis=bn_axis,
        name=name + '_preact_bn')(input_tensor)
    preact = layers.Activation(
        activation='relu',
        name=name + '_preact_relu')(preact)
    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters=4 * filters,
            kernel_size=1,
            strides=stride,
            name=name + '_0_conv')(preact)
    else:
        shortcut = input_tensor
    x = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=1,
        use_bias=False,
        name=name + '_1_conv')(preact)

    x = layers.BatchNormalization(
        axis=bn_axis,
        name=name + '_1_bn')(x)
    x = layers.Activation(
        activation='relu',
        name=name + '_1_relu')(x)
    x = layers.ZeroPadding2D(
        padding=1,
        name=name + '_2_pad')(x)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        use_bias=False,
        name=name + '_2_conv')(x)

    x = layers.BatchNormalization(
        axis=bn_axis,
        name=name + '_2_bn')(x)
    x = layers.Activation(
        activation='relu',
        name=name + '_2_relu')(x)
    x = layers.Conv2D(
        filters=4 * filters,
        kernel_size=1,
        strides=1,
        name=name + '_3_conv')(x)

    x = layers.Add(name=name + '_out')([shortcut, x])

    return x


def resnet_block(input_tensor, filters, size, stride=2, stage=None):
    """A set of stacked residual blocks.

    Arguments:
        input_tensor: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        size: integer, blocks in the stacked blocks.
        stride: default 2, stride of the first layer in the first block.
        stage: string, stack label.

    Returns:
        Output tensor for the stacked blocks.
    """
    base_name = 'stage' + str(stage)
    x = bottleneck_block(input_tensor, filters, conv_shortcut=True, stride=stride, name=base_name + '_block_0')
    for i in range(size - 1):
        x = bottleneck_block(x, filters, name=base_name + '_block_%d' % (i + 1))

    return x


def resnet50v2(num_classes,
               batch_size=None):

    """Instantiates the ResNet50V2 architecture.

    Args:
      num_classes: `int` number of classes for image classification.
      batch_size: Size of the batches for each step.

    Returns:
        A Keras model instance.
    """

    input_shape = (224, 224, 3)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    x = img_input

    if backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        bn_axis = 1
    else:  # channels_last
        bn_axis = -1

    x = layers.ZeroPadding2D(padding=3)(x)
    x = layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='conv1_conv')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = resnet_block(x, size=3, filters=64, stage=2, stride=1)
    x = resnet_block(x, size=4, filters=128, stage=3)
    x = resnet_block(x, size=6, filters=256, stage=4)
    x = resnet_block(x, size=3, filters=512, stage=5)

    x = layers.BatchNormalization(
        axis=bn_axis, name='post_bn')(x)
    x = layers.Activation('relu', name='post_relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        num_classes,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='fc1000')(x)

    # A softmax that is followed by the model loss must be done cannot be done
    # in float16 due to numeric issues. So we pass dtype=float32.
    x = layers.Activation('softmax', dtype='float32')(x)

    # Create model.
    return models.Model(img_input, x, name='resnet50v2')
