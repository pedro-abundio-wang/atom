#!/bin/bash

# Script to download and preprocess ImageNet Challenge 2012
# training and validation data set.
#
# The final output of this script are sharded TFRecord files containing
# serialized Example protocol buffers. See build_imagenet_data.py for
# details of how the Example protocol buffers contain the ImageNet data.
#
# The final output of this script appears as such:
#
#   data_dir/train-00000-of-01024
#   data_dir/train-00001-of-01024
#    ...
#   data_dir/train-00127-of-01024
#
# and
#
#   data_dir/validation-00000-of-00128
#   data_dir/validation-00001-of-00128
#   ...
#   data_dir/validation-00127-of-00128
#
# Note that this script may take several hours to run to completion. The
# conversion of the ImageNet data to TFRecords alone takes 2-3 hours depending
# on the speed of your machine. Please be patient.
set -e

if [ -z "$1" ]; then
  echo "usage imagenet_raw_to_tfrecord.sh [data dir]"
  exit
fi

# IMAGENET_HOME="/data/imagenet"
IMAGENET_HOME="$1"

# Download the ImageNet data.
LABELS_FILE="${IMAGENET_HOME}/imagenet_lsvrc_2015_synsets.txt"

# Note the locations of the train and validation data.
TRAIN_DIRECTORY="${IMAGENET_HOME}/train/"
VALIDATION_DIRECTORY="${IMAGENET_HOME}/validation-slim/"

# Preprocess the validation data by moving the images into the appropriate
# sub-directory based on the label (synset) of the image.
echo "Organizing the validation data into sub-directories."
PREPROCESS_VAL_SCRIPT="preprocess_imagenet_validation_data.py"
VAL_LABELS_FILE="${IMAGENET_HOME}/imagenet_2012_validation_synset_labels.txt"

python "${PREPROCESS_VAL_SCRIPT}" "${VALIDATION_DIRECTORY}" "${VAL_LABELS_FILE}"

# Convert the XML files for bounding box annotations into a single CSV.
echo "Extracting bounding box information from XML."
BOUNDING_BOX_SCRIPT="process_bounding_boxes.py"
BOUNDING_BOX_DIR="${IMAGENET_HOME}/bounding_boxes/"
BOUNDING_BOX_FILE="${IMAGENET_HOME}/imagenet_2012_bounding_boxes.csv"

LABELS_ANNOTATED="${BOUNDING_BOX_DIR}/*"
NUM_XML=$(find ${LABELS_ANNOTATED} -maxdepth 1 -type f | wc -l)
echo "Identified ${NUM_XML} bounding box annotations."

python "${BOUNDING_BOX_SCRIPT}" "${BOUNDING_BOX_DIR}" "${LABELS_FILE}" \
 | sort >"${BOUNDING_BOX_FILE}"
echo "Finished preprocessing the ImageNet data."

# Build the TFRecords version of the ImageNet data.
BUILD_SCRIPT="build_imagenet_data.py"
OUTPUT_DIRECTORY="${IMAGENET_HOME}/tfrecord-slim"
IMAGENET_METADATA_FILE="${IMAGENET_HOME}/imagenet_metadata.txt"

python "${BUILD_SCRIPT}" \
  --train_directory="${TRAIN_DIRECTORY}" \
  --validation_directory="${VALIDATION_DIRECTORY}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --imagenet_metadata_file="${IMAGENET_METADATA_FILE}" \
  --labels_file="${LABELS_FILE}" \
  --bounding_box_file="${BOUNDING_BOX_FILE}"
