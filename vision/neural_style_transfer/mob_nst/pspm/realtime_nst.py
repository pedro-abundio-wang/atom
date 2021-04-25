"""Perceptual Losses for Real-Time Style Transfer and Super-Resolution.

Related papers
- https://arxiv.org/abs/1603.08155

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import logging

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import applications
from tensorflow.keras import preprocessing


def preprocess_image(image_path):
    # Util function to open, resize and format pictures into appropriate tensors
    image = preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = applications.vgg19.preprocess_input(image)
    return tf.convert_to_tensor(image)


def deprocess_image(image):
    # Util function to convert a tensor into a valid image
    image = image.reshape((224, 224, 3))
    # Remove zero-center by mean pixel
    image[:, :, 0] += 103.939
    image[:, :, 1] += 116.779
    image[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    image = image[:, :, ::-1]
    image = np.clip(image, 0, 255).astype("uint8")
    return image


def residual_block(input_tensor,
                   filters):
    if backend.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=bn_axis)(x)

    x = layers.add([x, input_tensor])

    return x


def image_transformation_network():

    input_shape = (256, 256, 3)
    img_input = layers.Input(shape=input_shape)
    x = img_input

    if backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        channel_axis = 1
    else:  # channels_last
        channel_axis = -1

    x = layers.Conv2D(
            filters=32,
            kernel_size=9,
            strides=1,
            padding='same',
            kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)

    x = layers.Conv2DTranspose(
        filters=64,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(
        filters=32,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=3,
        kernel_size=9,
        strides=1,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = tf.nn.tanh(x) * 150 + 255. / 2

    # Create model.
    model = models.Model(img_input, x, name='image_transformation_network')

    model.summary()

    return model


def loss_network():
    # load pre-trained vgg model
    model = applications.vgg16.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3))

    for layer in model.layers:
        layer.trainable = False

    model.summary()

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in
    # VGG16 (as a dict).
    feature_extractor = models.Model(inputs=model.inputs, outputs=outputs_dict, name='vgg16_feature_extractor')

    return feature_extractor


def realtime_nst():

    input_shape = (256, 256, 3)
    img_input = layers.Input(shape=input_shape)
    x = img_input

    if backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)

    itn_model = image_transformation_network()
    loss_model = loss_network()

    gen_img = itn_model(x)
    features = loss_model(gen_img)

    model = models.Model(inputs=img_input, outputs=features, name='realtime_nst')

    model.summary()

    return model


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: Tensor of shape (1, height, width, channel) giving features for
      a single image.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (height * width * channel)

    Returns:
    - gram: Tensor of shape (channel, channel) giving the (optionally normalized)
      Gram matrices for the input image.
    """
    height, width, channel = features.shape
    feature_maps = tf.reshape(features, (height * width, channel))
    gram = tf.matmul(feature_maps, tf.transpose(feature_maps))
    if normalize:
        gram = tf.divide(gram, tf.cast(0.5 * height * width * channel, gram.dtype))

    return gram


def style_loss(style, combination):
    style_gram = gram_matrix(style, normalize=True)
    combination_gram = gram_matrix(combination, normalize=True)
    return tf.square(tf.norm(style_gram - combination_gram))


def content_loss(content, combination):
    return 0.5 * tf.square(tf.norm(combination - content))


def total_variation_loss(image):
    """
    Compute total variation loss.

    Inputs:
    - img: Tensor of shape (1, height, width, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    image = tf.squeeze(image)
    height, width, channel = image.shape

    img_col_start = tf.slice(image, [0, 0, 0], [height, width - 1, channel])
    img_col_end = tf.slice(image, [0, 1, 0], [height, width - 1, channel])
    img_row_start = tf.slice(image, [0, 0, 0], [height - 1, width, channel])
    img_row_end = tf.slice(image, [1, 0, 0], [height - 1, width, channel])
    return tf.reduce_sum(tf.square(img_col_end - img_col_start)) + tf.reduce_sum(tf.square(img_row_end - img_row_start))


def compute_loss(feature_extractor,
                 combination_image,
                 content_image,
                 style_image,
                 content_layer_name,
                 content_weight,
                 style_layer_names,
                 style_weights,
                 total_variation_weight):

    input_tensor = tf.concat(
        [content_image, style_image, combination_image], axis=0
    )

    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # content loss
    layer_features = features[content_layer_name]
    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(content_image_features, combination_features)

    # style loss
    if style_layer_names is not None:
        for i, layer_name in enumerate(style_layer_names):
            layer_features = features[layer_name]
            style_weight = style_weights[i]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            loss += style_weight * style_loss(style_features, combination_features)

    # total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss


def compute_loss_and_grads(feature_extractor,
                           combination_image,
                           content_image,
                           style_image,
                           content_layer_name,
                           content_weight,
                           style_layer_names,
                           style_weights,
                           total_variation_weight):
    """
    ## Add a tf.function decorator to loss & gradient computation
    To compile it, and thus make it fast.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(feature_extractor,
                            combination_image,
                            content_image,
                            style_image,
                            content_layer_name,
                            content_weight,
                            style_layer_names,
                            style_weights,
                            total_variation_weight)
    grads = tape.gradient(loss, combination_image)
    return loss, grads


def run(style_image_path,
        content_layer_name,
        content_weight,
        style_layer_names,
        style_weights,
        total_variation_weight,
        result_prefix):

    content_image_path = 'elephant.png'
    style_image_path = 'starry_night.jpg'

    content_image = preprocess_image(content_image_path)
    style_image = preprocess_image(style_image_path)
    combination_image = tf.Variable(preprocess_image(content_image_path))

    feature_extractor = realtime_nst()

    optimizer = optimizers.Adam(
        optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1.0, decay_steps=100, decay_rate=0.96
        )
    )

    iterations = 10000
    for i in range(1, iterations + 1):
        loss, grads = compute_loss_and_grads(
            feature_extractor,
            combination_image,
            content_image,
            style_image,
            content_layer_name,
            content_weight,
            style_layer_names,
            style_weights,
            total_variation_weight)
        grads = [(tf.clip_by_value(grad, -5.0, 5.0))
                 for grad in grads]
        optimizer.apply_gradients(zip(grads, feature_extractor.trainable_weights))
        if i % 100 == 0:
            logging.info("Iteration %d: loss=%.2f" % (i, loss))


def main(_):

    params = {
        'style_image_path': 'starry_night.jpg',
        'content_layer_name': 'block4_conv2',
        'content_weight': 0,
        'style_layer_names': [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ],
        'style_weights': [0.2, 0.2, 0.2, 0.2, 0.2],
        'total_variation_weight': 0,
        'result_prefix': 'style_reconstructions_block5_conv1'
    }

    run(**params)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)

