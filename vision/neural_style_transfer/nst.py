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

from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import applications
from tensorflow.keras import preprocessing


def load_model():
    """load pre-trained vgg model"""
    model = applications.vgg19.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3))
    model.summary()
    return model


def vis_img_filters(model):

    img_filters = [weight
                   for weight in model.weights
                   if 'kernel' in weight.name][0]

    f, axarr = plt.subplots(8, 8)

    for x in range(0, 8):
        kernel = img_filters[:, :, :, x].numpy()
        # Clip to [0, 1]
        kernel += 0.5
        kernel = np.clip(kernel, 0, 1)
        axarr[0, x].imshow(kernel, cmap='inferno')
        axarr[0, x].grid(False)

    plt.show()


def vis_feature_maps(model,
                     img_path='elephant.jpg',
                     convolution_number=1):
    """vis vgg feature-maps"""
    layer_outputs = [layer.output for layer in model.layers]
    feature_extractor = models.Model(
        inputs=model.input,
        outputs=layer_outputs)

    img = preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = applications.vgg19.preprocess_input(x)

    feature_maps = feature_extractor.predict(x)

    f, axarr = plt.subplots(1, 4)

    for x in range(0, 4):
        axarr[0, x].imshow(feature_maps[0, :, :, convolution_number], cmap='inferno')
        axarr[0, x].grid(False)

# content reconstruction


def random_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, 224, 224, 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25

# sytle reconstruction


def run():
    vgg_model = load_model()
    vis_img_filters(vgg_model)
    vis_feature_maps(vgg_model, img_path='elephant.jpg', convolution_number=1)


def main(_):
    run()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
