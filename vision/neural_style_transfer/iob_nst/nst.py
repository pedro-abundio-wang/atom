"""Neural Style Transfer model for Keras.

Related papers
- https://arxiv.org/abs/1508.06576

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import logging

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

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


def load_model():
    """load pre-trained vgg model"""
    model = applications.vgg19.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3))
    model.summary()
    return model


def vis_img_filters(model):
    # 1st conv filter weight
    img_filter_weight = [weight
                         for weight in model.weights
                         if 'kernel' in weight.name][0].numpy()
    # img_filter_num.shape = (in_channel, kernel_size, kernel_size, out_channel)
    img_filter_num = img_filter_weight.shape[-1]
    grid_size = int(np.ceil(np.sqrt(img_filter_num)))

    w_min, w_max = np.min(img_filter_weight), np.max(img_filter_weight)

    for i in range(img_filter_num):
        plt.subplot(grid_size, grid_size, i + 1)
        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (img_filter_weight[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')

    plt.savefig('vis/img_filter_weight.png')
    plt.clf()


def vis_feature_maps(model,
                     img_path='elephant.png'):
    """vis vgg feature maps"""
    # exclude input layer
    layer_outputs = [layer.output
                     for layer in model.layers
                     if not isinstance(layer, layers.InputLayer)]
    layer_names = [layer.name
                   for layer in model.layers
                   if not isinstance(layer, layers.InputLayer)]
    feature_extractor = models.Model(
        inputs=model.input,
        outputs=layer_outputs)

    img = preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = applications.vgg19.preprocess_input(x)

    feature_maps = feature_extractor.predict(x)

    for i, feature_map in enumerate(feature_maps):
        vis_feature_map_grid(feature_map, 'vis/%s_feature_map.png' % layer_names[i])


def vis_feature_map_grid(feature_map, save_path):
    # feature_map.shape = (1, height, width, channel)
    channel_size = feature_map.shape[-1]
    grid_size = int(np.ceil(np.sqrt(channel_size)))
    w_min, w_max = np.min(feature_map), np.max(feature_map)
    for i in range(channel_size):
        plt.subplot(grid_size, grid_size, i + 1)
        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (feature_map[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
    plt.savefig(save_path)
    plt.clf()


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


def compute_loss(model,
                 combination_image,
                 content_image,
                 style_image,
                 content_layer_name,
                 content_weight,
                 style_layer_names,
                 style_weights,
                 total_variation_weight):
    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in
    # VGG19 (as a dict).
    feature_extractor = models.Model(inputs=model.inputs, outputs=outputs_dict)

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
    for i, layer_name in enumerate(style_layer_names):
        layer_features = features[layer_name]
        style_weight = style_weights[i]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss += style_weight * style_loss(style_features, combination_features)

    # total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss


def compute_loss_and_grads(model,
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
        loss = compute_loss(model,
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


def neural_style_transfer(model,
                          content_image_path,
                          style_image_path,
                          content_layer_name,
                          content_weight,
                          style_layer_names,
                          style_weights,
                          total_variation_weight,
                          result_prefix,
                          init_random=True):
    # decay the learning rate by 0.96 every 100 steps.
    optimizer = optimizers.SGD(
        optimizers.schedules.ExponentialDecay(
            initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
        )
    )

    content_image = preprocess_image(content_image_path)
    style_image = preprocess_image(style_image_path)
    if init_random:
        combination_image = tf.Variable(tf.random.uniform((224, 224, 3)))
        combination_image = tf.expand_dims(combination_image, axis=0)
    else:
        combination_image = tf.Variable(preprocess_image(content_image))

    iterations = 4000
    for i in range(1, iterations + 1):
        loss, grads = compute_loss_and_grads(
            model,
            combination_image,
            content_image,
            style_image,
            content_layer_name,
            content_weight,
            style_layer_names,
            style_weights,
            total_variation_weight)
        optimizer.apply_gradients([(grads, combination_image)])
        if i % 100 == 0:
            logging.info("Iteration %d: loss=%.2f" % (i, loss))
            img = deprocess_image(combination_image.numpy())
            fname = "vis/%s_iteration_%d.png" % (result_prefix, i)
            preprocessing.image.save_img(fname, img)


def content_reconstructions(vgg_model):

    # block4_conv2_content_reconstructions
    params = {
        'content_image_path': 'elephant.png',
        'style_image_path': 'starry_night.jpg',
        'content_layer_name': 'block4_conv2',
        'content_weight': 1,
        'total_variation_weight': 1e-3,
        'result_prefix': 'block4_conv2_content_reconstructions',
        'init_random': True
    }

    neural_style_transfer(vgg_model, **params)

    # block2_conv2_content_reconstructions
    params = {
        'content_image_path': 'elephant.png',
        'style_image_path': 'starry_night.jpg',
        'content_layer_name': 'block2_conv2',
        'content_weight': 1,
        'total_variation_weight': 1e-3,
        'result_prefix': 'block2_conv2_content_reconstructions',
        'init_random': True
    }

    neural_style_transfer(vgg_model, **params)


def style_reconstructions(vgg_model):

    # block1_conv1_style_reconstructions
    params = {
        'content_image_path': 'elephant.png',
        'style_image_path': 'starry_night.jpg',
        'content_layer_name': 'block4_conv2',
        'content_weight': 0,
        'style_layer_names': [
            "block1_conv1",
        ],
        'style_weights': [1.0],
        'total_variation_weight': 1e-3,
        'result_prefix': 'block1_conv1_style_reconstructions',
        'init_random': True
    }

    neural_style_transfer(vgg_model, **params)

    # block3_conv1_style_reconstructions
    params = {
        'content_image_path': 'elephant.png',
        'style_image_path': 'starry_night.jpg',
        'content_layer_name': 'block4_conv2',
        'content_weight': 0,
        'style_layer_names': [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
        ],
        'style_weights': [0.33, 0.33, 0.33],
        'total_variation_weight': 1e-3,
        'result_prefix': 'block3_conv1_style_reconstructions',
        'init_random': True
    }

    neural_style_transfer(vgg_model, **params)

    # block5_conv1_style_reconstructions
    params = {
        'content_image_path': 'elephant.png',
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
        'total_variation_weight': 1e-3,
        'result_prefix': 'block5_conv1_style_reconstructions',
        'init_random': True
    }

    neural_style_transfer(vgg_model, **params)


def nst(vgg_model):
    # nst
    params = {
        'content_image_path': 'elephant.png',
        'style_image_path': 'starry_night.jpg',
        'content_layer_name': 'block4_conv2',
        'content_weight': 1e-3,
        'style_layer_names': [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ],
        'style_weights': [0.2, 0.2, 0.2, 0.2, 0.2],
        'total_variation_weight': 1e-3,
        'result_prefix': 'nst',
        'init_random': True
    }

    neural_style_transfer(vgg_model, **params)


def run():
    vgg_model = load_model()
    vis_img_filters(vgg_model)
    vis_feature_maps(vgg_model)

    content_reconstructions(vgg_model)
    style_reconstructions(vgg_model)
    nst(vgg_model)


def main(_):
    run()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
