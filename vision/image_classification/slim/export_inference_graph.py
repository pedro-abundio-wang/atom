r"""Saves out a GraphDef containing the architecture of the model.

To use it, run something like this, with a model name defined by slim:

bazel build tensorflow_models/research/slim:export_inference_graph
bazel-bin/tensorflow_models/research/slim/export_inference_graph \
--model_name=inception_v3 --output_file=/tmp/inception_v3_inf_graph.pb

If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:

bazel build tensorflow/python/tools:freeze_graph
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=/tmp/inception_v3_inf_graph.pb \
--input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
--input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
--output_node_names=InceptionV3/Predictions/Reshape_1

The output node names will vary depending on the model, but you can inspect and
estimate them using the summarize_graph tool:

bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=/tmp/inception_v3_inf_graph.pb

To run the resulting graph in C++, you can look at the label_image sample code:

bazel build tensorflow/examples/label_image:label_image
bazel-bin/tensorflow/examples/label_image/label_image \
--image=${HOME}/Pictures/flowers.jpg \
--input_layer=input \
--output_layer=InceptionV3/Predictions/Reshape_1 \
--graph=/tmp/frozen_inception_v3.pb \
--labels=/tmp/imagenet_slim_labels.txt \
--input_mean=0 \
--input_std=255

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import os

import tensorflow as tf

from vision.image_classification.slim.datasets import dataset_factory
from vision.image_classification.slim.nets import nets_factory


flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to save.')

flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

flags.DEFINE_integer(
    'image_size', None,
    'The image size to use, otherwise use the model default_image_size.')

flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

flags.DEFINE_string('dataset_name', 'imagenet',
                           'The name of the dataset to use with the model.')

flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')

flags.DEFINE_string(
    'dataset_dir', '', 'Directory to save intermediate dataset files to')

flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

flags.DEFINE_bool(
    'is_video_model', False, 'whether to use 5-D inputs for video model.')

flags.DEFINE_integer(
    'num_frames', None,
    'The number of frames to use. Only used if is_video_model is True.')

flags.DEFINE_bool('write_text_graphdef', False,
                         'Whether to write a text version of graphdef.')

flags.DEFINE_bool('use_grayscale', False,
                         'Whether to convert input images to grayscale.')

FLAGS = flags.FLAGS


def main(_):
  if not FLAGS.output_file:
    raise ValueError('You must supply the path to save to with --output_file')
  if FLAGS.is_video_model and not FLAGS.num_frames:
    raise ValueError(
        'Number of frames must be specified for video models with --num_frames')
  logging.set_verbosity(logging.INFO)
  with tf.compat.v1.Graph().as_default() as graph:
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train',
                                          FLAGS.dataset_dir)
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=FLAGS.is_training)
    image_size = FLAGS.image_size or network_fn.default_image_size
    num_channels = 1 if FLAGS.use_grayscale else 3
    if FLAGS.is_video_model:
      input_shape = [
          FLAGS.batch_size, FLAGS.num_frames, image_size, image_size,
          num_channels
      ]
    else:
      input_shape = [FLAGS.batch_size, image_size, image_size, num_channels]
    placeholder = tf.compat.v1.placeholder(name='input', dtype=tf.float32,
                                           shape=input_shape)
    network_fn(placeholder)

    if FLAGS.quantize:
      pass

    graph_def = graph.as_graph_def()
    if FLAGS.write_text_graphdef:
      tf.io.write_graph(
          graph_def,
          os.path.dirname(FLAGS.output_file),
          os.path.basename(FLAGS.output_file),
          as_text=True)
    else:
      with tf.io.gfile.GFile(FLAGS.output_file, 'wb') as f:
        f.write(graph_def.SerializeToString())


if __name__ == '__main__':
  app.run(main)
