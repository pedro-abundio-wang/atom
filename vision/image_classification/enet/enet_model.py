"""ENet model for Keras.

Related papers:
- https://arxiv.org/abs/1606.02147

"""

import tensorflow as tf
from tensorflow import keras

L2_WEIGHT_DECAY = 1e-4
PRELU_ALPHA = 0.25


def initial_block(input_tensor):

    if keras.backend.image_data_format() == 'channels_last':
        channel_axis = 3
    else:
        channel_axis = 1

    branch_conv = keras.layers.Conv2D(
        filters=13,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        name='initial_block_conv')(input_tensor)
    branch_conv = keras.layers.BatchNormalization(
        axis=channel_axis,
        name='initial_block_bn')(branch_conv)

    branch_pool = keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid')(input_tensor)

    x = keras.layers.Concatenate(axis=channel_axis)([branch_conv, branch_pool])
    x = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(x)

    return x


def bottleneck_block(input_tensor,
                     filters,
                     projection_ratio=4,
                     downsampling=False,
                     upsampling=False,
                     dilated=(1, 1),
                     asymmetric=False,
                     stage=None,
                     block=None):

    if keras.backend.image_data_format() == 'channels_last':
        channel_axis = 3
    else:
        channel_axis = 1

    x = input_tensor
    name_base = 'stage' + str(stage) + '_' + 'block' + str(block)
    reduced_depth = filters // projection_ratio

    if downsampling:
        x = keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid')(x)
        zero_padding = tf.zeros_like(x)
        x = keras.layers.Concatenate(axis=channel_axis)([x, zero_padding])

    if downsampling:
        shortcut = keras.layers.Conv2D(
            filters=reduced_depth,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
            name=name_base + '_conv_reduce')(x)
    else:
        shortcut = keras.layers.Conv2D(
            filters=reduced_depth,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
            name=name_base + '_conv_reduce')(x)
    shortcut = keras.layers.BatchNormalization(
        axis=channel_axis,
        name=name_base + '_bn_reduce')(shortcut)
    shortcut = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(shortcut)

    if asymmetric:
        shortcut = keras.layers.Conv2D(
            filters=reduced_depth,
            kernel_size=(5, 1),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
            name=name_base + '_conv')(shortcut)
        shortcut = keras.layers.Conv2D(
            filters=reduced_depth,
            kernel_size=(1, 5),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
            name=name_base + '_conv')(shortcut)
        shortcut = keras.layers.BatchNormalization(
            axis=channel_axis,
            name=name_base + '_bn')(shortcut)
        shortcut = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(shortcut)
    else:
        shortcut = keras.layers.Conv2D(
            filters=reduced_depth,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            dilation_rate=dilated,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
            name=name_base + '_conv')(shortcut)
        shortcut = keras.layers.BatchNormalization(
            axis=channel_axis,
            name=name_base + '_bn')(shortcut)
        shortcut = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(shortcut)

    shortcut = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        name=name_base + '_conv_expansion')(shortcut)
    shortcut = keras.layers.BatchNormalization(
        axis=channel_axis,
        name=name_base + '_bn_expansion')(shortcut)
    # TODO Regularizer - Spatial Dropout

    x = keras.layers.add([x, shortcut])
    x = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(x)

    return x


def enet_encoder(input_tensor):

    x = input_tensor

    # stage1
    x = bottleneck_block(x, filters=64, stage=1, block=1, downsampling=True)
    x = bottleneck_block(x, filters=64, stage=1, block=1)
    x = bottleneck_block(x, filters=64, stage=1, block=2)
    x = bottleneck_block(x, filters=64, stage=1, block=3)
    x = bottleneck_block(x, filters=64, stage=1, block=4)

    # stage2
    x = bottleneck_block(x, filters=128, stage=2, block=0, downsampling=True)
    x = bottleneck_block(x, filters=128, stage=2, block=1)
    x = bottleneck_block(x, filters=128, stage=2, block=2, dilated=(2, 2))
    x = bottleneck_block(x, filters=128, stage=2, block=3, asymmetric=True)
    x = bottleneck_block(x, filters=128, stage=2, block=4, dilated=(4, 4))
    x = bottleneck_block(x, filters=128, stage=2, block=5)
    x = bottleneck_block(x, filters=128, stage=2, block=6, dilated=(8, 8))
    x = bottleneck_block(x, filters=128, stage=2, block=7, asymmetric=True)
    x = bottleneck_block(x, filters=128, stage=2, block=8, dilated=(16, 16))

    # stage3
    x = bottleneck_block(x, filters=128, stage=3, block=1)
    x = bottleneck_block(x, filters=128, stage=3, block=2, dilated=(2, 2))
    x = bottleneck_block(x, filters=128, stage=3, block=3, asymmetric=True)
    x = bottleneck_block(x, filters=128, stage=3, block=4, dilated=(4, 4))
    x = bottleneck_block(x, filters=128, stage=3, block=5)
    x = bottleneck_block(x, filters=128, stage=3, block=6, dilated=(8, 8))
    x = bottleneck_block(x, filters=128, stage=3, block=7, asymmetric=True)
    x = bottleneck_block(x, filters=128, stage=3, block=8, dilated=(16, 16))

    return x


def enet(num_classes,
         batch_size=None):

    input_shape = (512, 512, 3)
    img_input = keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = img_input
    