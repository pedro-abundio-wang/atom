# Image Classification

* [Classifier Trainer](#classifier-trainer), a framework that uses the Keras
compile/fit methods for image classification models, including:
  * AlexNet (loss: 3.1398 - top_5_accuracy: 0.7995 - accuracy: 0.5642 - 30min/epoch)
  * ResNet18 (90min/epoch)
  * ResNet50 (5h/epoch)
  * GoogLeNet (20min/epoch)
  * ResNet18V2 (80min/epoch)
  * ResNet50V2 (4h/epoch)
  * InceptionV3
  * InceptionV4
  * Inception-ResNet
  * Vgg16 (9h/epoch)
  * WideResNet
  * ResNeXt
  * ShuffleNet
  * SqueezeNets (20min/epoch)
  * MobileNet
  * EfficientNet
  * Xception

### ImageNet preparation

[README](./imagenet/README.md)

### Running on multiple GPU hosts

You can also train these models on multiple hosts, each with GPUs, using
[tf.distribute.experimental.MultiWorkerMirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy).

## Classifier Trainer

The classifier trainer is a unified framework for running image classification
models using Keras's compile/fit methods. Experiments should be provided in the
form of YAML files, some examples are included within the configs/examples
folder. Please see [configs/examples](./configs/examples) for more example
configurations.

The provided configuration files use a per replica batch size and is scaled
by the number of devices. For instance, if `batch size` = 64, then for 1 GPU
the global batch size would be 64 * 1 = 64. For 8 GPUs, the global batch size
would be 64 * 8 = 512.

```bash
python classifier_trainer.py \
  --mode=train_and_eval \
  --model_type=$MODEL_NAME \
  --dataset=imagenet \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --config_file=configs/examples/$MODEL/imagenet/gpu.yaml
```
