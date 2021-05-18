"""Tests for export_inference_graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags

import tensorflow as tf

from tensorflow.python.platform import gfile
from vision.image_classification.slim import export_inference_graph


class ExportInferenceGraphTest(tf.test.TestCase):

  def testExportInferenceGraph(self):
    tmpdir = self.get_temp_dir()
    output_file = os.path.join(tmpdir, 'inception_v3.pb')
    FLAGS = flags.FLAGS
    FLAGS.output_file = output_file
    FLAGS.model_name = 'inception_v3'
    FLAGS.dataset_dir = tmpdir
    export_inference_graph.main(None)
    self.assertTrue(gfile.Exists(output_file))

if __name__ == '__main__':
  tf.test.main()
