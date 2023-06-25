# python3.7
"""Defines loss functions for Faster R-CNN training."""

from utils.dist_utils import ddp_sync
from .base_loss import BaseLoss

__all__ = ['FasterRCNNLoss']


class FasterRCNNLoss(BaseLoss):
    """Contains the class to compute losses for training Faster R-CNN."""

    def __init__(self, runner, loss_kwargs=None):
        """Initializes with models and arguments for computing losses."""

        runner.running_stats.add('Loss/FRCNN Box Reg',
                                 log_name='loss_box_reg',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/FRCNN Classifier',
                                 log_name='loss_classifier',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/FRCNN Objectness',
                                    log_name='loss_objectness',
                                    log_format='.3f',
                                    log_strategy='AVERAGE')
        runner.running_stats.add('Loss/FRCNN RPN Box Reg',
                                    log_name='loss_rpn_box_reg',
                                    log_format='.3f',
                                    log_strategy='AVERAGE')

    def frcnn_loss(self, runner, data, sync=True):
        """Computes loss for the whole model."""

        images, targets = data

        model = runner.ddp_models['frcnn']
        with ddp_sync(model, sync=sync):
            losses = model(images, targets)

        runner.running_stats.update({'Loss/FRCNN Box Reg': losses['loss_box_reg'].item()})
        runner.running_stats.update({'Loss/FRCNN Classifier': losses['loss_classifier'].item()})
        runner.running_stats.update({'Loss/FRCNN Objectness': losses['loss_objectness'].item()})
        runner.running_stats.update({'Loss/FRCNN RPN Box Reg': losses['loss_rpn_box_reg'].item()})

        total_loss = sum(loss for loss in losses.values())

        return total_loss
