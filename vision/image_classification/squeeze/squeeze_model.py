"""SqueezeNet model for Keras.

Related papers
- https://arxiv.org/abs/1602.07360

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import layers


def squeeze(input_tensor,
            base_name,
            s1x1):

    x = layers.Conv2D(
        filters=s1x1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        name=base_name + '_squeeze')(input_tensor)

    return x


def expand(input_tensor,
           base_name,
           e1x1,
           e3x3):

    if backend.image_data_format() == 'channels_last':
        channel_axis = -1
    else:
        channel_axis = 1

    expand1x1 = layers.Conv2D(
        filters=e1x1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        name=base_name + '_expand1x1')(input_tensor)

    expand3x3 = layers.Conv2D(
        filters=e3x3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        name=base_name + '_expand3x3')(input_tensor)

    x = layers.Concatenate(axis=channel_axis)([expand1x1, expand3x3])

    return x


def fire(input_tensor,
         name,
         s1x1,
         e1x1,
         e3x3):

    x = squeeze(input_tensor, name, s1x1)
    x = expand(x, name, e1x1, e3x3)
    return x


def squeezenet(num_classes,
               batch_size=None):

    """Instantiates the SqueezeNet architecture.

    Args:
        num_classes: `int` number of classes for image classification.
        batch_size: Size of the batches for each step.

    Returns:
        A Keras model instance.
    """

    input_shape = (224, 224, 3)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    x = img_input

    x = layers.Conv2D(
        filters=96,
        kernel_size=(7, 7),
        strides=(2, 2),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
        name='conv1')(x)
    x = layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='valid')(x)

    x = fire(x, name='fire2', s1x1=16, e1x1=64, e3x3=64)
    x = fire(x, name='fire3', s1x1=16, e1x1=64, e3x3=64)
    x = fire(x, name='fire4', s1x1=32, e1x1=128, e3x3=128)

    x = layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(2, 2))(x)

    x = fire(x, name='fire5', s1x1=32, e1x1=128, e3x3=128)
    x = fire(x, name='fire6', s1x1=48, e1x1=192, e3x3=192)
    x = fire(x, name='fire7', s1x1=48, e1x1=192, e3x3=192)
    x = fire(x, name='fire8', s1x1=64, e1x1=256, e3x3=256)

    x = layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(2, 2))(x)

    x = fire(x, name='fire9', s1x1=64, e1x1=256, e3x3=256)

    x = layers.Dropout(rate=0.5)(x)
    x = layers.Conv2D(
        filters=num_classes,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
        name='conv10')(x)

    x = layers.GlobalAveragePooling2D()(x)

    # A softmax that is followed by the model loss must be done cannot be done
    # in float16 due to numeric issues. So we pass dtype=float32.
    x = layers.Activation('softmax', dtype='float32')(x)

    # Create model.
    return models.Model(img_input, x, name='squeeze')
