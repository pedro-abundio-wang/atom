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


"""
## Compute the style transfer loss
First, we need to define 4 utility functions:
- `gram_matrix` (used to compute the style loss)
- The `style_loss` function, which keeps the generated image close to the local textures
of the style reference image
- The `content_loss` function, which keeps the high-level representation of the
generated image close to that of the base image
- The `total_variation_loss` function, a regularization loss which keeps the generated
image locally-coherent
"""

# The gram matrix of an image tensor (feature-wise outer product)


def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


# The "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = 224 * 224
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


# An auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image


def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))


# The 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def total_variation_loss(x):
    a = tf.square(
        x[:, : 224 - 1, : 224 - 1, :] - x[:, 1:, : 224 - 1, :]
    )
    b = tf.square(
        x[:, : 224 - 1, : 224 - 1, :] - x[:, : 224 - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))


def create_feature_extract():
    # Build a VGG19 model loaded with pre-trained ImageNet weights
    model = applications.vgg19.VGG19(weights="imagenet", include_top=False)

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in
    # VGG19 (as a dict).
    feature_extractor = models.Model(inputs=model.inputs, outputs=outputs_dict)

    return feature_extractor


def compute_loss(feature_extractor,
                 combination_image,
                 base_image,
                 style_reference_image,
                 content_layer_name,
                 content_weight,
                 style_layer_names,
                 style_weight,
                 total_variation_weight):

    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss


"""
## Add a tf.function decorator to loss & gradient computation
To compile it, and thus make it fast.
"""


@tf.function
def compute_loss_and_grads(feature_extractor,
                           combination_image,
                           base_image,
                           style_reference_image,
                           content_layer_name,
                           content_weight,
                           style_layer_names,
                           style_weight,
                           total_variation_weight):
    with tf.GradientTape() as tape:
        loss = compute_loss(feature_extractor,
                            combination_image,
                            base_image,
                            style_reference_image,
                            content_layer_name,
                            content_weight,
                            style_layer_names,
                            style_weight,
                            total_variation_weight)
    grads = tape.gradient(loss, combination_image)
    return loss, grads


"""
## The training loop
Repeatedly run vanilla gradient descent steps to minimize the loss, and save the
resulting image every 100 iterations.
We decay the learning rate by 0.96 every 100 steps.
"""


def neural_style_transfer(base_image_path,
                          style_reference_image_path,
                          result_prefix,
                          total_variation_weight,
                          style_weight,
                          content_weight,
                          style_layer_names,
                          content_layer_name):
    optimizer = optimizers.SGD(
        optimizers.schedules.ExponentialDecay(
            initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
        )
    )

    base_image = preprocess_image(base_image_path)
    style_reference_image = preprocess_image(style_reference_image_path)
    combination_image = tf.Variable(preprocess_image(base_image_path))

    feature_extract = create_feature_extract()

    iterations = 4000
    for i in range(1, iterations + 1):
        loss, grads = compute_loss_and_grads(
            feature_extract,
            combination_image,
            base_image,
            style_reference_image,
            content_layer_name,
            content_weight,
            style_layer_names,
            style_weight,
            total_variation_weight
        )
        optimizer.apply_gradients([(grads, combination_image)])
        if i % 100 == 0:
            print("Iteration %d: loss=%.2f" % (i, loss))
            img = deprocess_image(combination_image.numpy())
            fname = result_prefix + "_at_iteration_%d.png" % i
            preprocessing.image.save_img(fname, img)


def run():

    params = {
        'base_image_path': 'elephant.png',
        'style_reference_image_path': 'starry_night.jpg',
        'result_prefix': "paris_generated",
        'total_variation_weight': 1e-6,
        'style_weight': 1e-6,
        'content_weight': 2.5e-8,
        'style_layer_names': [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ],
        'content_layer_name': "block5_conv2",
    }
    neural_style_transfer(**params)


def main(_):
    run()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)

