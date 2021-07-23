from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Mapping

from absl import app
from absl import logging

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from vision.image_classification.alexnet import alexnet_model


def get_models() -> Mapping[str, tf.keras.Model]:
    """Returns the mapping from model type name to Keras model."""
    return {
        'alexnet': alexnet_model.alexnet
    }


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

    plt.savefig('img_filter_weight.png')
    plt.clf()


def load_model():
    model_dir = '/data/models/alexnet/imagenet'
    model_params = {'batch_size': None, 'num_classes': 1000}
    model = get_models()['alexnet'](**model_params)

    logging.info('Load from checkpoint is enabled.')
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    logging.info('latest_checkpoint: %s', latest_checkpoint)
    if not latest_checkpoint:
        logging.info('No checkpoint detected.')

    logging.info('Checkpoint file %s found and restoring from '
                 'checkpoint', latest_checkpoint)
    model.load_weights(latest_checkpoint)
    logging.info('Completed loading from checkpoint.')
    return model


def run():
    model = load_model()
    vis_img_filters(model)


def main(_):
    run()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)



