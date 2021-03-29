"""ResNet50 model for Keras.

Related papers/blogs:
- https://arxiv.org/abs/1512.03385

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import layers
from tensorflow.python.keras import backend
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers

L2_WEIGHT_DECAY = 1e-4


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block):
    """The identity block is the block that has no conv layer at shortcut.

    Args:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names

    Returns:
      Output tensor for the block.
    """

    filters1, filters2, filters3 = filters

    if backend.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters1,
        (1, 1),
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters3,
        (1, 1),
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)

    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well

    Args:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the second conv layer in the block.

    Returns:
      Output tensor for the block.
    """

    filters1, filters2, filters3 = filters

    if backend.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters1,
        (1, 1),
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters3,
        (1, 1),
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(
        filters3,
        (1, 1),
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


def resnet50(num_classes,
             batch_size=None):
    """Instantiates the ResNet50 architecture.

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

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = layers.Conv2D(
        64, (7, 7),
        strides=(2, 2),
        padding='valid',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='conv1')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

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
    return models.Model(img_input, x, name='resnet50')
