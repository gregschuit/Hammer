# python3.7
"""Contains the runner for Faster R-CNN."""

from .base_runner import BaseRunner

__all__ = ['FasterRCNNRunner']


class FasterRCNNRunner(BaseRunner):
    """Defines the runner for Faster R-CNN."""

    def build_models(self):
        super().build_models()

    def build_loss(self):
        super().build_loss()

    def train_step(self, data):
        self.models['frcnn'].requires_grad_(True)
        self.zero_grad_optimizer('frcnn')
        total_loss = self.loss.frcnn_loss(self, data, sync=True)
        total_loss.backward()
        self.step_optimizer('frcnn')
