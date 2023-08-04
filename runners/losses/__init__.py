# python3.7
"""Collects all loss functions."""

from .faster_rcnn_loss import FasterRCNNLoss
from .stylegan_loss import StyleGANLoss
from .stylegan2_loss import StyleGAN2Loss
from .stylegan3_loss import StyleGAN3Loss
from .stylegan2_region_based_loss import StyleGAN2RegionBasedLoss

__all__ = ['build_loss']

_LOSSES = {
    'FasterRCNNLoss': FasterRCNNLoss,
    'StyleGANLoss': StyleGANLoss,
    'StyleGAN2Loss': StyleGAN2Loss,
    'StyleGAN3Loss': StyleGAN3Loss,
    'StyleGAN2RegionBasedLoss': StyleGAN2RegionBasedLoss,
}


def build_loss(runner, loss_type, **kwargs):
    """Builds a loss based on its class type.

    Args:
        runner: The runner on which the loss is built.
        loss_type: Class type to which the loss belongs, which is case
            sensitive.
        **kwargs: Additional arguments to build the loss.

    Raises:
        ValueError: If the `loss_type` is not supported.
    """
    if loss_type not in _LOSSES:
        raise ValueError(f'Invalid loss type: `{loss_type}`!\n'
                         f'Types allowed: {list(_LOSSES)}.')
    return _LOSSES[loss_type](runner, **kwargs)
