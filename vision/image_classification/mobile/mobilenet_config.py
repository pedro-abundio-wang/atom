# Lint as: python3
# ==============================================================================
"""Configuration definitions for AlexNet losses, learning rates, and optimizers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Mapping

import dataclasses

from vision.image_classification.configs import base_configs


_LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1e-1, 5), (1e-2, 20), (1e-3, 40), (1e-4, 60), (1e-5, 80)
]

_LR_BOUNDARIES = list(p[1] for p in _LR_SCHEDULE[1:])
_LR_MULTIPLIERS = list(p[0] for p in _LR_SCHEDULE)
_LR_WARMUP_EPOCHS = _LR_SCHEDULE[0][1]


@dataclasses.dataclass
class MobileNetModelConfig(base_configs.ModelConfig):
    """Configuration for the model."""
    name: str = 'MobileNet'
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
            warmup_epochs=_LR_WARMUP_EPOCHS,
            boundaries=_LR_BOUNDARIES,
            multipliers=_LR_MULTIPLIERS))
