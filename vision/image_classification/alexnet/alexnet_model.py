"""AlexNet model for Keras.

Related papers
- http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras import layers

L2_WEIGHT_DECAY = 5e-4


def alexnet(num_classes,
            batch_size=None):
    """Instantiates the AlexNet architecture.

    Args:
        num_classes: `int` number of classes for image classification.
        batch_size: Size of the batches for each step.

    Returns:
        A Keras model instance.
    """

    input_shape = (227, 227, 3)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    x = img_input

    if backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        channel_axis = 1
    else:  # channels_last
        channel_axis = -1

    # stage1
    x = layers.Conv2D(
        filters=96,
        kernel_size=(11, 11),
        strides=(4, 4),
        padding='valid',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='stage1_conv')(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        name='stage1_bn')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    # stage2
    x = layers.Conv2D(
        filters=256,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='stage2_conv')(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        name='stage2_bn')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    # stage3
    x = layers.Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='stage3_conv')(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        name='stage3_bn')(x)
    x = layers.Activation('relu')(x)

    # stage4
    x = layers.Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='stage4_conv')(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        name='stage4_bn')(x)
    x = layers.Activation('relu')(x)

    # stage5
    x = layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='stage5_conv')(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        name='stage5_bn')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = layers.Flatten()(x)

    # fc
    x = layers.Dense(
        units=4096,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='fc')(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        name='bn')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(rate=0.5, name='dropout')(x)

    x = layers.Dense(
        units=4096,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        name='fc_')(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
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
    return models.Model(img_input, x, name='alexnet')
