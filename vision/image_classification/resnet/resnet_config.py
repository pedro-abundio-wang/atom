# Lint as: python3
# ==============================================================================
"""Configuration definitions for ResNet losses, learning rates, and optimizers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Mapping

import dataclasses

from vision.image_classification.configs import base_configs

_RESNET_LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (0.2, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]
_RESNET_LR_BOUNDARIES = list(p[1] for p in _RESNET_LR_SCHEDULE[1:])
_RESNET_LR_MULTIPLIERS = list(p[0] for p in _RESNET_LR_SCHEDULE)
_RESNET_LR_WARMUP_EPOCHS = _RESNET_LR_SCHEDULE[0][1]


@dataclasses.dataclass
class ResNet18ModelConfig(base_configs.ModelConfig):
    """Configuration for the ResNet18 model."""
    name: str = 'ResNet18'
    num_classes: int = 1000
    model_params: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {
        'num_classes': 1000,
        'batch_size': None,
    })
    loss: base_configs.LossConfig = base_configs.LossConfig(
        name='sparse_categorical_crossentropy')
    optimizer: base_configs.OptimizerConfig = base_configs.OptimizerConfig(
        name='momentum',
        decay=0.9,
        epsilon=0.001,
        momentum=0.9,
        moving_average_decay=None)
    learning_rate: base_configs.LearningRateConfig = (
        base_configs.LearningRateConfig(
            name='piecewise_constant_with_warmup',
            examples_per_epoch=1281167,
            warmup_epochs=_RESNET_LR_WARMUP_EPOCHS,
            boundaries=_RESNET_LR_BOUNDARIES,
            multipliers=_RESNET_LR_MULTIPLIERS))


@dataclasses.dataclass
class ResNet50ModelConfig(base_configs.ModelConfig):
    """Configuration for the ResNet50 model."""
    name: str = 'ResNet50'
    num_classes: int = 1000
    model_params: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {
        'num_classes': 1000,
        'batch_size': None,
    })
    loss: base_configs.LossConfig = base_configs.LossConfig(
        name='sparse_categorical_crossentropy')
    optimizer: base_configs.OptimizerConfig = base_configs.OptimizerConfig(
        name='momentum',
        decay=0.9,
        epsilon=0.001,
        momentum=0.9,
        moving_average_decay=None)
    learning_rate: base_configs.LearningRateConfig = (
        base_configs.LearningRateConfig(
            name='piecewise_constant_with_warmup',
            examples_per_epoch=1281167,
            warmup_epochs=_RESNET_LR_WARMUP_EPOCHS,
            boundaries=_RESNET_LR_BOUNDARIES,
            multipliers=_RESNET_LR_MULTIPLIERS))
