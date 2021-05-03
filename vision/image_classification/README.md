# Image Classification

* [Classifier Trainer](#classifier-trainer) a framework that uses the Keras
compile/fit methods for image classification models, including:
  * AlexNet
    * Total params: 62,416,616
    * Trainable params: 62,397,480
    * Non-trainable params: 19,136
    * batch_size=256, dtype=float32, epochs=90, 30min/epoch, GTX1060
    * loss: 3.4680 - accuracy: 0.5954 - top_5_accuracy: 0.7324
    * val_loss: 3.0642 - val_accuracy: 0.5754 - val_top_5_accuracy: 0.8072
  * ResNet18
    * Total params: 11,708,328
    * Trainable params: 11,698,600
    * Non-trainable params: 9,728
    * batch_size=256, dtype=mixed_float16, epochs=90, 10min/epoch, TeslaV100
    * loss: 2.7720 - accuracy: 0.6427 - top_5_accuracy: 0.8429
    * val_loss: 2.6050 - val_accuracy: 0.6700 - val_top_5_accuracy: 0.8755
  * ResNet50
    * Total params: 25,636,712
    * Trainable params: 25,583,592
    * Non-trainable params: 53,120
    * batch_size=256, dtype=mixed_float16, epochs=90, 20min/epoch, TeslaV100
    * loss: 2.4390 - accuracy: 0.7356 - top_5_accuracy: 0.8999
    * val_loss: 2.3760 - val_accuracy: 0.7373 - val_top_5_accuracy: 0.9157
  * ResNet18V2
    * Total params: 11,700,648
    * Trainable params: 11,692,840
    * Non-trainable params: 7,808
    * batch_size=256, dtype=mixed_float16, epochs=90, 10min/epoch, TeslaV100
    * loss: 2.8414 - accuracy: 0.5777 - top_5_accuracy: 0.7967
    * val_loss: 2.6130 - val_accuracy: 0.6187 - val_top_5_accuracy: 0.8403
  * ResNet50V2
    * Total params: 25,613,800
    * Trainable params: 25,568,360
    * Non-trainable params: 45,440
    * batch_size=256, dtype=mixed_float16, epochs=90, 20min/epoch, TeslaV100
    * loss: 2.5177 - accuracy: 0.6495 - top_5_accuracy: 0.8491 
    * val_loss: 2.3493 - val_accuracy: 0.6777 - val_top_5_accuracy: 0.8817
  * GoogLeNet
    * Total params: 7,027,672
    * Trainable params: 7,013,112
    * Non-trainable params: 14,560
    * batch_size=256, dtype=mixed_float16, epochs=90, 15min/epoch, TeslaV100
    * loss: 2.8186 - accuracy: 0.6290 - top_5_accuracy: 0.8353
    * val_loss: 2.5143 - val_accuracy: 0.6864 - val_top_5_accuracy: 0.8859
  * Inception
  * Inception-ResNet
  * Xception
  * Vgg16
    * Total params: 138,407,208
    * Trainable params: 138,382,376
    * Non-trainable params: 24,832
  * Vgg19
    * Total params: 143,722,024
    * Trainable params: 143,694,632
    * Non-trainable params: 27,392
  * WideResNet
  * ResNeXt
  * ShuffleNet
  * SqueezeNets
    * Total params: 1,248,424
    * Trainable params: 1,248,424
    * Non-trainable params: 0
  * MobileNet
    * Total params: 4,253,864
    * Trainable params: 4,231,976
    * Non-trainable params: 21,888
  * EfficientNet

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
