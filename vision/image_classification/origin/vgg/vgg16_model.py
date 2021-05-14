"""Vgg16 model for Keras.

Related papers
- https://arxiv.org/abs/1409.1556

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras import layers

L2_WEIGHT_DECAY = 1e-4


def vgg_block(input_tensor,
              size,
              filters,
              stage):

    x = input_tensor

    if backend.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    for i in range(size - 1):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name='stage' + stage + 'block_conv%d' % (i + 1))(x)
        x = layers.BatchNormalization(
            axis=bn_axis,
            name='stage' + stage + 'block_bn%d' % (i + 1))(x)
        x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same')(x)

    return x


def vgg16(num_classes,
          batch_size=None):

    """Instantiates the Vgg16 architecture.

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

    x = vgg_block(x, size=3, filters=64, stage='1')
    x = vgg_block(x, size=3, filters=128, stage='2')
    x = vgg_block(x, size=4, filters=256, stage='3')
    x = vgg_block(x, size=4, filters=512, stage='4')
    x = vgg_block(x, size=4, filters=512, stage='5')

    x = layers.Flatten()(x)

    x = layers.Dense(
        units=4096,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='fc')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        name='bn')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(rate=0.5, name='dropout')(x)

    x = layers.Dense(
        units=4096,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='fc_')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        name='bn_')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(rate=0.5, name='dropout_')(x)

    x = layers.Dense(
        units=num_classes,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='fc_score')(x)

    # A softmax that is followed by the model loss must be done
    # cannot be done in float16 due to numeric issues.
    # So we pass dtype=float32.
    x = layers.Activation('softmax', dtype='float32')(x)

    # Create model.
    return models.Model(img_input, x, name='vgg16')
