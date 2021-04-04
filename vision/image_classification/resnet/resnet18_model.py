"""ResNet18 model for Keras.

Related papers
- https://arxiv.org/abs/1512.03385

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras import layers

L2_WEIGHT_DECAY = 1e-4


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block):

    """The identity block is the block that has no conv layer at shortcut.

    Arguments:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: integer, filters of the layer.
      stage: integer, current stage label, used for generating layer names
      block: current block label, used for generating layer names

    Returns:
      Output tensor for the block.
    """

    if backend.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2b')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)

    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=2):

    """A block that has a conv layer at shortcut.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well

    Arguments:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: integer, filters of the layer.
      stage: integer, current stage label, used for generating layer names
      block: current block label, used for generating layer names
      strides: Strides for the first conv layer in the block.

    Returns:
      Output tensor for the block.
    """

    if backend.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2b')(x)

    shortcut = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=strides,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    return x


def resnet_block(input_tensor,
                 size,
                 kernel_size,
                 filters,
                 stage,
                 conv_strides=2):
    """A block which applies conv followed by multiple identity blocks.

    Arguments:
      input_tensor: input tensor
      size: integer, number of constituent conv/identity building blocks.
        A conv block is applied once, followed by (size - 1) identity blocks.
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: integer, filters of the layer.
      stage: integer, current stage label, used for generating layer names
      conv_strides: Strides for the first conv layer in the block.

    Returns:
      Output tensor after applying conv and identity blocks.
    """

    x = conv_block(input_tensor, kernel_size, filters, stage, 'block_0', conv_strides)
    for i in range(size - 1):
        x = identity_block(x, kernel_size, filters, stage, 'block_%d' % (i + 1))
    return x


def resnet18(num_classes,
             batch_size=None):

    """Instantiates the ResNet architecture.

    Arguments:
      num_classes: optional number of classes to classify images into
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

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='valid',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='conv1')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = resnet_block(x, size=2, kernel_size=3, filters=64, stage=2, conv_strides=1)
    x = resnet_block(x, size=2, kernel_size=3, filters=128, stage=3)
    x = resnet_block(x, size=2, kernel_size=3, filters=256, stage=4)
    x = resnet_block(x, size=2, kernel_size=3, filters=512, stage=5)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        units=num_classes,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='fc1000')(x)

    # A softmax that is followed by the model loss must be done
    # cannot be done in float16 due to numeric issues.
    # So we pass dtype=float32.
    x = layers.Activation('softmax', dtype='float32')(x)

    # Create model.
    return models.Model(img_input, x, name='resnet18')
