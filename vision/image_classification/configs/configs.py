# Lint as: python3
# ==============================================================================
"""Configuration utils for image classification experiments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataclasses

from vision.image_classification import dataset_factory
from vision.image_classification.configs import base_configs
from vision.image_classification.alexnet import alexnet_config
from vision.image_classification.resnet import resnet_config


@dataclasses.dataclass()
class AlexNetImagenetConfig(base_configs.ExperimentConfig):
    """Base configuration to train alexnet on ImageNet."""
    export: base_configs.ExportConfig = base_configs.ExportConfig()
    runtime: base_configs.RuntimeConfig = base_configs.RuntimeConfig()
    train_dataset: dataset_factory.DatasetConfig = \
        dataset_factory.ImageNetConfig(split='train',
                                       one_hot=False,
                                       mean_subtract=True,
                                       standardize=True)
    validation_dataset: dataset_factory.DatasetConfig = \
        dataset_factory.ImageNetConfig(split='validation',
                                       one_hot=False,
                                       mean_subtract=True,
                                       standardize=True)
    train: base_configs.TrainConfig = base_configs.TrainConfig(
        resume_checkpoint=True,
        epochs=90,
        steps=None,
        callbacks=base_configs.CallbacksConfig(enable_checkpoint_and_export=True,
                                               enable_tensorboard=True),
        metrics=['accuracy', 'top_5'],
        time_history=base_configs.TimeHistoryConfig(log_steps=100),
        tensorboard=base_configs.TensorboardConfig(track_lr=True,
                                                   write_model_weights=False))
    evaluation: base_configs.EvalConfig = base_configs.EvalConfig(
        epochs_between_evals=1,
        steps=None)
    model: base_configs.ModelConfig = alexnet_config.AlexNetModelConfig()


@dataclasses.dataclass
class ResNet18ImagenetConfig(base_configs.ExperimentConfig):
    """Base configuration to train resnet-18 on ImageNet."""
    export: base_configs.ExportConfig = base_configs.ExportConfig()
    runtime: base_configs.RuntimeConfig = base_configs.RuntimeConfig()
    train_dataset: dataset_factory.DatasetConfig = \
        dataset_factory.ImageNetConfig(split='train',
                                       one_hot=False,
                                       mean_subtract=True,
                                       standardize=True)
    validation_dataset: dataset_factory.DatasetConfig = \
        dataset_factory.ImageNetConfig(split='validation',
                                       one_hot=False,
                                       mean_subtract=True,
                                       standardize=True)
    train: base_configs.TrainConfig = base_configs.TrainConfig(
        resume_checkpoint=True,
        epochs=90,
        steps=None,
        callbacks=base_configs.CallbacksConfig(enable_checkpoint_and_export=True,
                                               enable_tensorboard=True),
        metrics=['accuracy', 'top_5'],
        time_history=base_configs.TimeHistoryConfig(log_steps=100),
        tensorboard=base_configs.TensorboardConfig(track_lr=True,
                                                   write_model_weights=False))
    evaluation: base_configs.EvalConfig = base_configs.EvalConfig(
        epochs_between_evals=1,
        steps=None)
    model: base_configs.ModelConfig = resnet_config.ResNet18ModelConfig()


@dataclasses.dataclass
class ResNet50ImagenetConfig(base_configs.ExperimentConfig):
    """Base configuration to train resnet-50 on ImageNet."""
    export: base_configs.ExportConfig = base_configs.ExportConfig()
    runtime: base_configs.RuntimeConfig = base_configs.RuntimeConfig()
    train_dataset: dataset_factory.DatasetConfig = \
        dataset_factory.ImageNetConfig(split='train',
                                       one_hot=False,
                                       mean_subtract=True,
                                       standardize=True)
    validation_dataset: dataset_factory.DatasetConfig = \
        dataset_factory.ImageNetConfig(split='validation',
                                       one_hot=False,
                                       mean_subtract=True,
                                       standardize=True)
    train: base_configs.TrainConfig = base_configs.TrainConfig(
        resume_checkpoint=True,
        epochs=90,
        steps=None,
        callbacks=base_configs.CallbacksConfig(enable_checkpoint_and_export=True,
                                               enable_tensorboard=True),
        metrics=['accuracy', 'top_5'],
        time_history=base_configs.TimeHistoryConfig(log_steps=100),
        tensorboard=base_configs.TensorboardConfig(track_lr=True,
                                                   write_model_weights=False))
    evaluation: base_configs.EvalConfig = base_configs.EvalConfig(
        epochs_between_evals=1,
        steps=None)
    model: base_configs.ModelConfig = resnet_config.ResNet50ModelConfig()


def get_config(model: str, dataset: str) -> base_configs.ExperimentConfig:
    """Given model and dataset names, return the ExperimentConfig."""
    dataset_model_config_map = {
        'imagenet': {
            'alexnet': AlexNetImagenetConfig(),
            'resnet18': ResNet18ImagenetConfig(),
            'resnet50': ResNet50ImagenetConfig(),
        }
    }
    try:
        return dataset_model_config_map[dataset][model]
    except KeyError:
        if dataset not in dataset_model_config_map:
            raise KeyError('Invalid dataset received. Received: {}. Supported '
                           'datasets include: {}'.format(dataset, ', '.join(dataset_model_config_map.keys())))
        raise KeyError('Invalid model received. Received: {}. Supported models for'
                       '{} include: {}'.format(model, dataset, ', '.join(dataset_model_config_map[dataset].keys())))
