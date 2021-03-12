"""ResNet model for Keras.
Related papers
- https://arxiv.org/abs/1512.03385
"""

from tensorflow.keras import backend
from tensorflow.keras import initializers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from vision.image_classification.resnet import imagenet_preprocessing


def l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
    return regularizers.l2(l2_weight_decay) if use_l2_regularizer else None


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   use_l2_regularizer=True,
                   batch_norm_decay=0.99,
                   batch_norm_epsilon=1e-5,
                   training=None):

    """The identity block is the block that has no conv layer at shortcut.

    Arguments:
      input_tensor: input tensor
      kernel_size: default (3, 3), the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 2 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: current block label, used for generating layer names
      use_l2_regularizer: whether to use L2 regularizer on Conv layer.
      batch_norm_decay: Moment of batch norm layers.
      batch_norm_epsilon: Epsilon of batch borm layers.
      training: Only used if training keras model with Estimator.  In other
        scenarios it is handled automatically.

    Returns:
      Output tensor for the block.
    """
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters=filters1,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2a')(x, training=training)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=filters2,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2b')(x, training=training)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)

    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               use_l2_regularizer=True,
               batch_norm_decay=0.99,
               batch_norm_epsilon=1e-5,
               training=None):

    """A block that has a conv layer at shortcut.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well

    Arguments:
      input_tensor: input tensor
      kernel_size: default (3, 3), the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: current block label, used for generating layer names
      strides: Strides for the first conv layer in the block.
      use_l2_regularizer: whether to use L2 regularizer on Conv layer.
      batch_norm_decay: Moment of batch norm layers.
      batch_norm_epsilon: Epsilon of batch borm layers.
      training: Only used if training keras model with Estimator.  In other
        scenarios it is handled automatically.

    Returns:
      Output tensor for the block.
    """
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters=filters1,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2a')(x, training=training)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=filters2,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2b')(x, training=training)

    shortcut = layers.Conv2D(
        filters=filters2,
        kernel_size=(1, 1),
        strides=strides,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '1')(shortcut, training=training)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    return x


def resnet_block(input_tensor,
                 size,
                 kernel_size,
                 filters,
                 stage,
                 conv_strides=(2, 2),
                 use_l2_regularizer=True,
                 batch_norm_decay=0.99,
                 batch_norm_epsilon=1e-5,
                 training=None):
    """A block which applies conv followed by multiple identity blocks.

    Arguments:
      input_tensor: input tensor
      size: integer, number of constituent conv/identity building blocks.
        A conv block is applied once, followed by (size - 1) identity blocks.
      kernel_size: default (3, 3), the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 2 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      conv_strides: Strides for the first conv layer in the block.
      use_l2_regularizer: whether to use L2 regularizer on Conv layer.
      batch_norm_decay: Moment of batch norm layers.
      batch_norm_epsilon: Epsilon of batch borm layers.
      training: Only used if training keras model with Estimator.  In other
        scenarios it is handled automatically.

    Returns:
      Output tensor after applying conv and identity blocks.
    """

    x = conv_block(input_tensor, kernel_size, filters,
                   stage, 'block_0', conv_strides,
                   use_l2_regularizer, batch_norm_decay,
                   batch_norm_epsilon, training=training)

    for i in range(size - 1):
        x = identity_block(x, kernel_size, filters,
                           stage, 'block_%d' % (i + 1),
                           use_l2_regularizer, batch_norm_decay,
                           batch_norm_epsilon, training=training)
    return x


def resnet18(num_classes,
             batch_size=None,
             use_l2_regularizer=True,
             rescale_inputs=False,
             batch_norm_decay=0.9,
             batch_norm_epsilon=1e-5,
             training=None):

    """Instantiates the ResNet architecture.

    Arguments:
      num_classes: optional number of classes to classify images into
      batch_size: Size of the batches for each step.
      use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
      rescale_inputs: whether to rescale inputs from 0 to 1.
      batch_norm_decay: Moment of batch norm layers.
      batch_norm_epsilon: Epsilon of batch norm layers.
      training: Only used if training keras model with Estimator.  In other
      scenarios it is handled automatically.

    Returns:
      A Keras model instance.
    """

    input_shape = (224, 224, 3)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    if rescale_inputs:
        # Hub image modules expect inputs in the range [0, 1]. This rescales these
        # inputs to the range expected by the trained model.
        x = layers.Lambda(
            lambda image: image * 255.0 - backend.constant(
                imagenet_preprocessing.CHANNEL_MEANS,
                shape=[1, 1, 3],
                dtype=image.dtype),
            name='rescale')(img_input)
    else:
        x = img_input

    if backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        bn_axis = 1
    else:  # channels_last
        bn_axis = 3

    block_config = dict(
        use_l2_regularizer=use_l2_regularizer,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon,
        training=training)

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name='conv1')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = resnet_block(x, size=2, kernel_size=(3, 3), filters=[64, 64],
                     stage=2, conv_strides=(1, 1), **block_config)

    x = resnet_block(x, size=2, kernel_size=(3, 3), filters=[128, 128],
                     stage=3, conv_strides=(2, 2), **block_config)

    x = resnet_block(x, size=2, kernel_size=(3, 3), filters=[256, 256],
                     stage=4, conv_strides=(2, 2), **block_config)

    x = resnet_block(x, size=2, kernel_size=(3, 3), filters=[512, 512],
                     stage=4, conv_strides=(2, 2), **block_config)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        units=num_classes,
        kernel_initializer=initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        bias_regularizer=l2_regularizer(use_l2_regularizer),
        name='fc1000')(x)

    # A softmax that is followed by the model loss must be done
    # cannot be done in float16 due to numeric issues.
    # So we pass dtype=float32.
    x = layers.Activation('softmax', dtype='float32')(x)

    # Create model.
    return models.Model(img_input, x, name='resnet18')


def bottleneck_identity_block(input_tensor,
                              kernel_size,
                              filters,
                              stage,
                              block,
                              use_l2_regularizer=True,
                              batch_norm_decay=0.9,
                              batch_norm_epsilon=1e-5):
    """The identity block is the block that has no conv layer at shortcut.

    Args:
      input_tensor: input tensor
      kernel_size: default (3, 3), the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: current block label, used for generating layer names
      use_l2_regularizer: whether to use L2 regularizer on Conv layer.
      batch_norm_decay: Moment of batch norm layers.
      batch_norm_epsilon: Epsilon of batch borm layers.

    Returns:
      Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters=filters1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=filters2,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=filters3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def bottleneck_conv_block(input_tensor,
                          kernel_size,
                          filters,
                          stage,
                          block,
                          strides=(2, 2),
                          use_l2_regularizer=True,
                          batch_norm_decay=0.9,
                          batch_norm_epsilon=1e-5):
    """A block that has a conv layer at shortcut.

    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well

    Args:
      input_tensor: input tensor
      kernel_size: default (3, 3), the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: current block label, used for generating layer names
      strides: Strides for the second conv layer in the block.
      use_l2_regularizer: whether to use L2 regularizer on Conv layer.
      batch_norm_decay: Moment of batch norm layers.
      batch_norm_epsilon: Epsilon of batch borm layers.

    Returns:
      Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters=filters1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=filters2,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=filters3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(
        filters=filters3,
        kernel_size=(1, 1),
        strides=strides,
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def bottleneck_resnet_block(input_tensor,
                            size,
                            kernel_size,
                            filters,
                            stage,
                            conv_strides=(2, 2),
                            use_l2_regularizer=True,
                            batch_norm_decay=0.99,
                            batch_norm_epsilon=1e-5):
    """A block which applies conv followed by multiple identity blocks.

    Arguments:
      input_tensor: input tensor
      size: integer, number of constituent conv/identity building blocks.
        A conv block is applied once, followed by (size - 1) identity blocks.
      kernel_size: default (3, 3), the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      conv_strides: Strides for the first conv layer in the block.
      use_l2_regularizer: whether to use L2 regularizer on Conv layer.
      batch_norm_decay: Moment of batch norm layers.
      batch_norm_epsilon: Epsilon of batch borm layers.

    Returns:
      Output tensor after applying conv and identity blocks.
    """
    x = bottleneck_conv_block(input_tensor, kernel_size, filters,
                              stage, 'block_0', conv_strides,
                              use_l2_regularizer, batch_norm_decay,
                              batch_norm_epsilon)

    for i in range(size - 1):
        x = bottleneck_identity_block(x, kernel_size, filters,
                                      stage, 'block_%d' % (i + 1),
                                      use_l2_regularizer, batch_norm_decay,
                                      batch_norm_epsilon)
    return x


def resnet50(num_classes,
             batch_size=None,
             use_l2_regularizer=True,
             rescale_inputs=False,
             batch_norm_decay=0.9,
             batch_norm_epsilon=1e-5):
    """Instantiates the ResNet50 architecture.
    Args:
      num_classes: `int` number of classes for image classification.
      batch_size: Size of the batches for each step.
      use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
      rescale_inputs: whether to rescale inputs from 0 to 1.
      batch_norm_decay: Moment of batch norm layers.
      batch_norm_epsilon: Epsilon of batch norm layers.
    Returns:
        A Keras model instance.
    """
    input_shape = (224, 224, 3)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    if rescale_inputs:
        # Hub image modules expect inputs in the range [0, 1]. This rescales these
        # inputs to the range expected by the trained model.
        x = layers.Lambda(
            lambda image: image * 255.0 - backend.constant(
                imagenet_preprocessing.CHANNEL_MEANS,
                shape=[1, 1, 3],
                dtype=image.dtype),
            name='rescale')(img_input)
    else:
        x = img_input

    if backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        bn_axis = 1
    else:  # channels_last
        bn_axis = 3

    block_config = dict(
        use_l2_regularizer=use_l2_regularizer,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon)

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        name='conv1')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = bottleneck_resnet_block(x, size=3, kernel_size=(3, 3), filters=[64, 64, 256],
                                stage=2, conv_strides=(1, 1), **block_config)

    x = bottleneck_resnet_block(x, size=4, kernel_size=(3, 3), filters=[128, 128, 512],
                                stage=3, conv_strides=(2, 2), **block_config)

    x = bottleneck_resnet_block(x, size=6, kernel_size=(3, 3), filters=[256, 256, 1024],
                                stage=4, conv_strides=(2, 2), **block_config)

    x = bottleneck_resnet_block(x, size=3, kernel_size=(3, 3), filters=[512, 512, 2048],
                                stage=5, conv_strides=(2, 2), **block_config)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        units=num_classes,
        kernel_initializer=initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=l2_regularizer(use_l2_regularizer),
        bias_regularizer=l2_regularizer(use_l2_regularizer),
        name='fc1000')(x)

    # A softmax that is followed by the model loss must be done
    # cannot be done in float16 due to numeric issues.
    # So we pass dtype=float32.
    x = layers.Activation('softmax', dtype='float32')(x)

    # Create model.
    return models.Model(img_input, x, name='resnet50')

