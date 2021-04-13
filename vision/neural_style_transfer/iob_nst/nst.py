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
    """vis vgg feature-maps"""
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
    vis_feature_maps(vgg_model)


def main(_):
    run()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
