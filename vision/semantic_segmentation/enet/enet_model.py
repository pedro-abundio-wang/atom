"""ENet model for Keras.

Related papers:
- https://arxiv.org/abs/1606.02147

"""

import tensorflow as tf
from tensorflow import keras

import vision.semantic_segmentation.layers as layers

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


def bottleneck(input_tensor,
               filters,
               stage,
               block,
               drop_rate=0.1,
               dilation_rate=(1, 1),
               projection_ratio=4):

    if keras.backend.image_data_format() == 'channels_last':
        channel_axis = -1
    else:
        channel_axis = 1

    name_base = 'stage' + str(stage) + '_' + 'block' + str(block)
    reduced_depth = filters // projection_ratio

    shortcut = keras.layers.Conv2D(
        filters=reduced_depth,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        name=name_base + '_conv_reduce')(input_tensor)
    shortcut = keras.layers.BatchNormalization(
        axis=channel_axis,
        name=name_base + '_bn_reduce')(shortcut)
    shortcut = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(shortcut)
    
    shortcut = keras.layers.Conv2D(
        filters=reduced_depth,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        dilation_rate=dilation_rate,
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
    shortcut = keras.layers.SpatialDropout2D(rate=drop_rate)(shortcut)

    x = keras.layers.add([input_tensor, shortcut])
    x = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(x)

    return x


def upsampling(input_tensor,
               in_filters,
               out_filters,
               mask,
               stage,
               block,
               drop_rate=0.1,
               projection_ratio=4):

    if keras.backend.image_data_format() == 'channels_last':
        channel_axis = -1
    else:
        channel_axis = 1

    name_base = 'stage' + str(stage) + '_' + 'block' + str(block)
    reduced_depth = in_filters // projection_ratio

    # spatial convolution/upsampling max pooling
    x = keras.layers.Conv2D(
        filters=out_filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        name=name_base + '_conv_upsampling_reduce')(input_tensor)
    x = layers.MaxUnpooling2D()(pool_size=(2, 2))([x, mask])

    shortcut = keras.layers.Conv2DTranspose(
        filters=reduced_depth,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        output_padding=(0, 0),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        name=name_base + '_conv_reduce')(input_tensor)
    shortcut = keras.layers.BatchNormalization(
        axis=channel_axis,
        name=name_base + '_bn_reduce')(shortcut)
    shortcut = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(shortcut)

    shortcut = keras.layers.Conv2DTranspose(
        filters=reduced_depth,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        output_padding=(1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        name=name_base + '_conv_upsampling')(shortcut)
    shortcut = keras.layers.BatchNormalization(
        axis=channel_axis,
        name=name_base + '_bn')(shortcut)
    shortcut = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(shortcut)

    shortcut = keras.layers.Conv2DTranspose(
        filters=out_filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        output_padding=(0, 0),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        name=name_base + '_conv_expansion')(shortcut)
    shortcut = keras.layers.BatchNormalization(
        axis=channel_axis,
        name=name_base + '_bn_expansion')(shortcut)
    shortcut = keras.layers.SpatialDropout2D(rate=drop_rate)(shortcut)

    x = keras.layers.add([x, shortcut])
    x = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(x)

    return x


def downsampling(input_tensor,
                 in_filters,
                 out_filters,
                 stage,
                 block,
                 drop_rate=0.1,
                 projection_ratio=4):

    if keras.backend.image_data_format() == 'channels_last':
        channel_axis = -1
    else:
        channel_axis = 1

    name_base = 'stage' + str(stage) + '_' + 'block' + str(block)
    reduced_depth = in_filters // projection_ratio

    x, mask = layers.MaxPoolingWithArgmax2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same')(input_tensor)
    zero_padding = tf.zeros_like(x)
    x = keras.layers.Concatenate(axis=channel_axis)([x, zero_padding])

    shortcut = keras.layers.Conv2D(
        filters=reduced_depth,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        name=name_base + '_conv_reduce')(input_tensor)
    shortcut = keras.layers.BatchNormalization(
        axis=channel_axis,
        name=name_base + '_bn_reduce')(shortcut)
    shortcut = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(shortcut)

    shortcut = keras.layers.Conv2D(
        filters=reduced_depth,
        kernel_size=(3, 3),
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

    shortcut = keras.layers.Conv2D(
        filters=out_filters,
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
    shortcut = keras.layers.SpatialDropout2D(rate=drop_rate)(shortcut)

    x = keras.layers.add([x, shortcut])
    x = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(x)

    return x, mask


def asymmetric(input_tensor,
               filters,
               stage,
               block,
               drop_rate=0.1,
               projection_ratio=4):

    if keras.backend.image_data_format() == 'channels_last':
        channel_axis = -1
    else:
        channel_axis = 1

    name_base = 'stage' + str(stage) + '_' + 'block' + str(block)
    reduced_depth = filters // projection_ratio

    shortcut = keras.layers.Conv2D(
        filters=reduced_depth,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        name=name_base + '_conv_reduce')(input_tensor)
    shortcut = keras.layers.BatchNormalization(
        axis=channel_axis,
        name=name_base + '_bn_reduce')(shortcut)
    shortcut = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(shortcut)

    shortcut = keras.layers.Conv2D(
        filters=reduced_depth,
        kernel_size=(5, 1),
        strides=(1, 1),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        name=name_base + '_conv5x1')(shortcut)
    shortcut = keras.layers.Conv2D(
        filters=reduced_depth,
        kernel_size=(1, 5),
        strides=(1, 1),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        name=name_base + '_conv1x5')(shortcut)
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
    shortcut = keras.layers.SpatialDropout2D(rate=drop_rate)(shortcut)

    x = keras.layers.add([input_tensor, shortcut])
    x = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(x)

    return x


def enet_encoder(input_tensor):

    x = input_tensor

    # stage1
    x, mask1 = downsampling(x, in_filters=16, out_filters=64, stage=1, block=1, drop_rate=0.01)
    x = bottleneck(x, filters=64, stage=1, block=1, drop_rate=0.01)
    x = bottleneck(x, filters=64, stage=1, block=2, drop_rate=0.01)
    x = bottleneck(x, filters=64, stage=1, block=3, drop_rate=0.01)
    x = bottleneck(x, filters=64, stage=1, block=4, drop_rate=0.01)

    # stage2
    x, mask2 = downsampling(x, in_filters=64, out_filters=128, stage=2, block=0)
    x = bottleneck(x, filters=128, stage=2, block=1)
    x = bottleneck(x, filters=128, stage=2, block=2, dilation_rate=(2, 2))
    x = asymmetric(x, filters=128, stage=2, block=3)
    x = bottleneck(x, filters=128, stage=2, block=4, dilation_rate=(4, 4))
    x = bottleneck(x, filters=128, stage=2, block=5)
    x = bottleneck(x, filters=128, stage=2, block=6, dilation_rate=(8, 8))
    x = asymmetric(x, filters=128, stage=2, block=7)
    x = bottleneck(x, filters=128, stage=2, block=8, dilation_rate=(16, 16))

    # stage3
    x = bottleneck(x, filters=128, stage=3, block=1)
    x = bottleneck(x, filters=128, stage=3, block=2, dilation_rate=(2, 2))
    x = asymmetric(x, filters=128, stage=3, block=3)
    x = bottleneck(x, filters=128, stage=3, block=4, dilation_rate=(4, 4))
    x = bottleneck(x, filters=128, stage=3, block=5)
    x = bottleneck(x, filters=128, stage=3, block=6, dilation_rate=(8, 8))
    x = asymmetric(x, filters=128, stage=3, block=7)
    x = bottleneck(x, filters=128, stage=3, block=8, dilation_rate=(16, 16))

    return x, mask1, mask2


def enet_decoder(input_tensor,
                 masks,
                 num_classes):

    mask1, mask2 = masks

    # stage4
    x = upsampling(input_tensor, in_filters=128, out_filters=64, mask=mask2, stage=4, block=0)
    x = bottleneck(x, filters=64, stage=4, block=1)
    x = bottleneck(x, filters=64, stage=4, block=2)

    # stage5
    x = upsampling(x, in_filters=64, out_filters=16, mask=mask1, stage=5, block=0)
    x = bottleneck(x, filters=16, stage=5, block=1)

    # full conv
    x = keras.layers.ConvTranspose2d(
        filters=num_classes,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        output_padding=(1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        name='full_conv')(x)

    return x


def enet(num_classes,
         batch_size=None):

    input_shape = (512, 512, 3)
    img_input = keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = img_input

    x = initial_block(x)
    x, mask1, mask2 = enet_encoder(x)
    x = enet_decoder(x, [mask1, mask2], num_classes)

    # Create model.
    return keras.models.Model(img_input, x, name='enet')

