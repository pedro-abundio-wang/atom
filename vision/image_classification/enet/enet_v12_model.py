"""ENet V12 model for Keras.

Related papers/blogs:
- https://culurciello.github.io/tech/2016/06/20/training-enet.html

"""

import tensorflow as tf
from tensorflow import keras

L2_WEIGHT_DECAY = 1e-4
PRELU_ALPHA = 0.25


def initial_block(input_tensor):
    """ENet initial block
    :param input_tensor: input tensor
    :return: initial block tensor
    """

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
               in_filters,
               out_filters,
               stage,
               block,
               drop_rate=0.1,
               dilation_rate=(1, 1),
               projection_ratio=4):
    """ENet bottleneck block
    :param input_tensor: input tensor
    :param in_filters: integer, the input filters of conv layers at shortcut path
    :param out_filters: integer, the out filters of conv layers at shortcut path
    :param stage: integer, current stage label, used for generating layer names
    :param block: integer, current block label, used for generating layer names
    :param drop_rate: spatial dropout rate
    :param dilation_rate: shortcut conv dilation rate
    :param projection_ratio: integer, the projection ratio of conv layers at shortcut path
    :return: bottleneck block tensor
    """

    if keras.backend.image_data_format() == 'channels_last':
        channel_axis = -1
    else:
        channel_axis = 1

    name_base = 'stage' + str(stage) + '_' + 'block' + str(block)
    reduced_depth = in_filters // projection_ratio

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

    x = keras.layers.add([input_tensor, shortcut])
    x = keras.layers.PReLU(alpha_initializer=PRELU_ALPHA)(x)

    return x


def downsampling(input_tensor,
                 in_filters,
                 out_filters,
                 stage,
                 block,
                 drop_rate=0.1,
                 projection_ratio=4):
    """ENet downsampling bottleneck block
    :param input_tensor: input tensor
    :param in_filters: integer, the input filters of conv layers at shortcut path
    :param out_filters: integer, the out filters of conv layers at shortcut path
    :param stage: integer, current stage label, used for generating layer names
    :param block: integer, current block label, used for generating layer names
    :param drop_rate: spatial dropout rate
    :param projection_ratio: integer, the projection ratio of conv layers at shortcut path
    :return: downsampling bottleneck block tensor
    """

    if keras.backend.image_data_format() == 'channels_last':
        channel_axis = -1
    else:
        channel_axis = 1

    name_base = 'stage' + str(stage) + '_' + 'block' + str(block)
    reduced_depth = in_filters // projection_ratio

    x = keras.layers.MaxPool2D(
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

    return x


def enet_encoder(input_tensor):
    """ENet encoder
    :param input_tensor: input tensor
    :return: encoder tensor
    """

    x = input_tensor
    masks = []

    # stage1
    x = downsampling(x, in_filters=16, out_filters=64, stage=1, block=0)
    x = bottleneck(x, in_filters=64, out_filters=128, stage=1, block=1)
    x = bottleneck(x, in_filters=128, out_filters=128, stage=1, block=2)

    # stage2
    x = downsampling(x, in_filters=128, out_filters=256, stage=2, block=0)
    x = bottleneck(x, in_filters=256, out_filters=256, stage=2, block=1)
    x = bottleneck(x, in_filters=256, out_filters=256, stage=2, block=2, dilation_rate=(2, 2))

    # stage3
    x = downsampling(x, in_filters=256, out_filters=512, stage=3, block=0)
    x = bottleneck(x, in_filters=512, out_filters=512, stage=3, block=1)
    x = bottleneck(x, in_filters=512, out_filters=512, stage=3, block=2, dilation_rate=(4, 4))

    # stage4
    x = downsampling(x, in_filters=512, out_filters=1024, stage=4, block=0)
    x = bottleneck(x, in_filters=1024, out_filters=1024, stage=4, block=1)
    x = bottleneck(x, in_filters=1024, out_filters=1024, stage=4, block=2, dilation_rate=(8, 8))

    x = keras.layers.GlobalAveragePooling2D()(x)

    return x, masks


def enet(num_classes,
         batch_size=None):
    """Instantiates the ENet architecture.

    Args:
      num_classes: `int` number of classes for semantic segmentation.
      batch_size: Size of the batches for each step.

    Returns:
        A Keras model instance.
    """

    input_shape = (224, 224, 3)
    img_input = keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = img_input

    x = initial_block(x)
    x, masks = enet_encoder(x)

    # classifier
    x = keras.layers.Dense(
        units=num_classes,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        bias_regularizer=keras.regularizers.l2(L2_WEIGHT_DECAY),
        name='fc1000')(x)

    # A softmax that is followed by the model loss must be done
    # cannot be done in float16 due to numeric issues.
    # So we pass dtype=float32.
    x = keras.layers.Activation('softmax', dtype='float32')(x)

    # Create model.
    return keras.models.Model(img_input, x, name='enet_v12')

