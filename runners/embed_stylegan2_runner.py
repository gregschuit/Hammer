# python3.7
"""Contains the runner for StyleGAN2 with GAN Inversion via Encoder."""

from copy import deepcopy

from .stylegan2_runner import StyleGAN2Runner

__all__ = ['EmbedStyleGAN2Runner']


class EmbedStyleGAN2Runner(StyleGAN2Runner):
    """Defines the runner for StyleGAN2 with Encoder-based GAN Inversion"""

    # def build_loss(self):
    #     super().build_loss()
    #     self.running_stats.add('Misc/Gs Beta',
    #                            log_name='Gs_beta',
    #                            log_format='.4f',
    #                            log_strategy='CURRENT')

    def _train_step_reconstruction(self, data):
        # Update encoder and generator.
        self.models['discriminator'].requires_grad_(False)
        self.models['generator'].requires_grad_(True)
        self.models['encoder'].requires_grad_(True)

        # Update with reconstruction loss.
        reconstruction_loss = self.loss.reconstruction_loss(self, data, sync=True)
        self.zero_grad_optimizer('generator')
        self.zero_grad_optimizer('encoder')
        reconstruction_loss.backward()
        self.step_optimizer('generator')
        self.step_optimizer('encoder')

    def train_step(self, data):
        self._train_step_generator(data)
        self._train_step_reconstruction(data)
        self._train_step_discriminator(data)
        self._smooth_generator()
