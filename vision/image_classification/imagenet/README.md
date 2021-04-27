# Tools for preparing datasets

**Image-Net preparing**

processes imagenet into `TFRecords`.

```bash
# `local_scratch_dir` will be where the TFRecords are stored.`
python imagenet_to_tfrecord.py \
  --raw_data_dir=/data/imagenet \
  --local_scratch_dir=/data/imagenet/tfrecord
```

**Image-Net with existing .tar files**

Utilizes already downloaded .tar files of the images

```bash
export IMAGENET_HOME=/data/imagenet

# Setup folders
mkdir -p $IMAGENET_HOME/train
mkdir -p $IMAGENET_HOME/validation

# Extract validation and training
tar xf ILSVRC2012_img_train.tar -C $IMAGENET_HOME/train
tar xf ILSVRC2012_img_val.tar -C $IMAGENET_HOME/validation

# Extract and then delete individual training tar files. This can be pasted
# directly into a bash command-line or create a file and execute.

cd $IMAGENET_HOME/train

for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  tar xf $f -C $d
done

cd $IMAGENET_HOME # Move back to the base folder

# [Optional] Delete tar files if desired as they are not needed
rm $IMAGENET_HOME/train/*.tar

# Download labels file.
wget -O $IMAGENET_HOME/synset_labels.txt \
https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/imagenet_2012_validation_synset_labels.txt

# Process the files. The TFRecords will end up in the --local_scratch_dir. 
python imagenet_to_tfrecord.py \
  --raw_data_dir=$IMAGENET_HOME \
  --local_scratch_dir=$IMAGENET_HOME/tfrecord
```