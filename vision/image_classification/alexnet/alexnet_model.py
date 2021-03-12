"""AlexNet model for Keras.
Related papers
- http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
"""

import tensorflow as tf


def l2_regularizer(use_l2_regularizer=True, l2_weight_decay=5e-4):
    return tf.keras.regularizers.l2(l2_weight_decay) if use_l2_regularizer else None


def alexnet(num_classes,
            batch_size=None,
            use_l2_regularizer=True,
            batch_norm_epsilon=1e-6):
    """Instantiates the AlexNet architecture.
    Args:
      num_classes: `int` number of classes for image classification.
      batch_size: Size of the batches for each step.
      use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
      batch_norm_epsilon: Epsilon of batch norm layers.
    Returns:
        A Keras model instance.
    """

    input_shape = (227, 227, 3)
    img_input = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = img_input

    if tf.keras.backend.image_data_format() == 'channels_first':
        x = tf.keras.layers.Permute((3, 1, 2))(x)
        bn_axis = 1
    else:  # channels_last
        bn_axis = -1

    # stage1
    x = tf.keras.layers.Conv2D(
        filters=96,
        kernel_size=(11, 11),
        strides=(4, 4),
        padding='valid',
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name='stage1_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis,
        epsilon=batch_norm_epsilon,
        name='stage1_bn')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    # stage2
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name='stage2_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis,
        epsilon=batch_norm_epsilon,
        name='stage2_bn')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    # stage3
    x = tf.keras.layers.Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name='stage3_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis,
        epsilon=batch_norm_epsilon,
        name='stage3_bn')(x)
    x = tf.keras.layers.Activation('relu')(x)

    # stage4
    x = tf.keras.layers.Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name='stage4_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis,
        epsilon=batch_norm_epsilon,
        name='stage4_bn')(x)
    x = tf.keras.layers.Activation('relu')(x)

    # stage5
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name='stage5_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis,
        epsilon=batch_norm_epsilon,
        name='stage5_bn')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)

    # fc
    x = tf.keras.layers.Dense(
        units=4096,
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name='fc')(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis,
        epsilon=batch_norm_epsilon,
        name='bn')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(rate=0.5, name='dropout')(x)

    x = tf.keras.layers.Dense(
        units=4096,
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name='fc_')(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis,
        epsilon=batch_norm_epsilon,
        name='bn_')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(rate=0.5, name='dropout_')(x)

    x = tf.keras.layers.Dense(
        units=num_classes,
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name='fc_score')(x)

    # A softmax that is followed by the model loss must be done
    # cannot be done in float16 due to numeric issues.
    # So we pass dtype=float32.
    x = tf.keras.layers.Activation('softmax', dtype='float32')(x)

    # Create model.
    return tf.keras.models.Model(img_input, x, name='alexnet')
