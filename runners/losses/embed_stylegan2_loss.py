# python3.7
"""Defines loss functions for StyleGAN2 training."""

import numpy as np

import torch
import torch.nn.functional as F

from third_party.stylegan2_official_ops import conv2d_gradfix
from utils.dist_utils import ddp_sync
from .base_loss import BaseLoss

__all__ = ['EmbedStyleGAN2Loss']

# TODO:

# Primero debería implementarlo sin region_based loss.
# Entonces:
# 1. CHECK Parametrizar el uso de region_based loss.
# 2. Programar el uso del encoder para embeber las imágenes reales.

# Opción 1:
# Tomar la mitad del batch, y usar imagenes reales.
# Pasar las imágenes reales por el encoder y por el clasificador para obtener w.
# Pasar z por las FF layers para obtener w.
# Juntar ambos w y seguir con el flujo normal.

# Para la opción 1.
# En concreto:
# 1. Obtener imágenes reales. Esta viene en la variable _data.
# 2. Cargar el encoder. Por ahora puede ser una variable global?
# 3. Calcular cuántas imágenes equivalen a la mitad del batch (o a un cuarto).
# 4. Pasar las imágenes por el encoder.
# 5. Probar con z o con w?
# 6. Si elijo w, tengo que editar la condicionalidad para que siempre condicione
#    con w, y no con z.


class EmbedStyleGAN2Loss(BaseLoss):
    """Contains the class to compute losses for training StyleGAN2.

    Basically, this class contains the computation of adversarial loss for both
    generator and discriminator, perceptual path length regularization for
    generator, and gradient penalty as the regularization for discriminator.
    """

    def __init__(self, runner, d_loss_kwargs=None, g_loss_kwargs=None):
        """Initializes with models and arguments for computing losses."""

        if runner.enable_amp:
            raise NotImplementedError('StyleGAN2 loss does not support '
                                      'automatic mixed precision training yet.')

        # Setting for discriminator loss.
        self.d_loss_kwargs = d_loss_kwargs or dict()
        # Loss weight for gradient penalty on real images.
        self.r1_gamma = self.d_loss_kwargs.get('r1_gamma', 10.0)
        # How often to perform gradient penalty regularization.
        self.r1_interval = self.d_loss_kwargs.get('r1_interval', 16)

        if self.r1_interval is None or self.r1_interval <= 0:
            self.r1_interval = 1
            self.r1_gamma = 0.0
        self.r1_interval = int(self.r1_interval)
        assert self.r1_gamma >= 0.0
        runner.running_stats.add('Loss/D Fake',
                                 log_name='loss_d_fake',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/D Real',
                                 log_name='loss_d_real',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        if self.r1_gamma > 0.0:
            runner.running_stats.add('Loss/Real Gradient Penalty',
                                     log_name='loss_gp',
                                     log_format='.1e',
                                     log_strategy='AVERAGE')

        # Settings for generator loss.
        self.g_loss_kwargs = g_loss_kwargs or dict()
        # Factor to shrink the batch size for path length regularization.
        self.pl_batch_shrink = int(self.g_loss_kwargs.get('pl_batch_shrink', 2))
        # Loss weight for perceptual path length regularization.
        self.pl_weight = self.g_loss_kwargs.get('pl_weight', 2.0)
        # Decay factor for perceptual path length regularization.
        self.pl_decay = self.g_loss_kwargs.get('pl_decay', 0.01)
        # How often to perform perceptual path length regularization.
        self.pl_interval = self.g_loss_kwargs.get('pl_interval', 4)

        if self.pl_interval is None or self.pl_interval <= 0:
            self.pl_interval = 1
            self.pl_weight = 0.0
        self.pl_interval = int(self.pl_interval)
        assert self.pl_batch_shrink >= 1
        assert self.pl_weight >= 0.0
        assert 0.0 <= self.pl_decay <= 1.0
        runner.running_stats.add('Loss/G',
                                 log_name='loss_g',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        if self.pl_weight > 0.0:
            runner.running_stats.add('Loss/Path Length Penalty',
                                     log_name='loss_pl',
                                     log_format='.1e',
                                     log_strategy='AVERAGE')
            self.pl_mean = torch.zeros((), device=runner.device)

        # Log loss settings.
        runner.logger.info('gradient penalty (D regularizer):', indent_level=1)
        runner.logger.info(f'r1_gamma: {self.r1_gamma}', indent_level=2)
        runner.logger.info(f'r1_interval: {self.r1_interval}', indent_level=2)
        runner.logger.info('perceptual path length penalty (G regularizer):',
                           indent_level=1)
        runner.logger.info(f'pl_batch_shrink: {self.pl_batch_shrink}',
                           indent_level=2)
        runner.logger.info(f'pl_weight: {self.pl_weight}', indent_level=2)
        runner.logger.info(f'pl_decay: {self.pl_decay}', indent_level=2)
        runner.logger.info(f'pl_interval: {self.pl_interval}', indent_level=2)
    
        # Reconstruction losses
        runner.running_stats.add('Loss/Recon. Image L1',
                                 log_name='loss_recon_img_l1',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/Recon. Class Logits KL',
                                 log_name='loss_recon_logits_kl',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/Recon. Latent z L1',
                                 log_name='loss_recon_latent_z_l1',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/Recon. Image LPIPS',
                                 log_name='loss_recon_img_lpips',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')

    @staticmethod
    def run_G_from_latents(runner, latents, labels=None, sync=True):
        G = runner.ddp_models['generator']
        G_kwargs = runner.model_kwargs_train['generator']
        with ddp_sync(G, sync=sync):
            return G(z=latents, label=labels, **G_kwargs)
    
    @staticmethod
    def run_G_from_latents_w(runner, latents, labels=None, sync=True):
        G = runner.ddp_models['generator']
        G_kwargs = runner.model_kwargs_train['generator']
        with ddp_sync(G, sync=sync):
            return G(z=None, label=labels, **G_kwargs, w=latents)

    def run_G(self, runner, batch_size=None, sync=True, requires_grad=False):
        """Forwards generator.

        NOTE: The flag `requires_grad` sets whether to compute the gradient for
            latent z. When computing the `pl_penalty` with part of the generator
            frozen (e.g., mapping network), this flag should be set to `True` to
            retain the computation graph.
        """
        # Prepare latent codes and labels.
        batch_size = batch_size or runner.batch_size
        latent_dim = runner.models['generator'].latent_dim
        label_dim = runner.models['generator'].label_dim
        latents = torch.randn((batch_size, *latent_dim),
                              device=runner.device,
                              requires_grad=requires_grad)
        labels = None
        if label_dim > 0:
            rnd_labels = torch.randint(
                0, label_dim, (batch_size,), device=runner.device)
            labels = F.one_hot(rnd_labels, num_classes=label_dim)

        return self.run_G_from_latents(runner, latents, labels, sync=sync)

    @staticmethod
    def run_E(runner, images, _labels=None, sync=True):
        """Encodes images into latent codes."""
        # Forward encoder.
        E = runner.ddp_models['encoder']
        E_kwargs = runner.model_kwargs_train['encoder']
        with ddp_sync(E, sync=sync):
            return E(images, **E_kwargs)

    @staticmethod
    def run_C(runner, images, _labels=None, sync=True):
        """Classifies images."""
        # Forward classifier.
        C = runner.ddp_models['classifier']
        C_kwargs = runner.model_kwargs_train['classifier']
        with ddp_sync(C, sync=sync):
            return C(images, **C_kwargs)

    @staticmethod
    def run_D(runner, images, labels, sync=True):
        """Forwards discriminator."""
        # Augment the images.
        images = runner.augment(images, **runner.augment_kwargs)

        # Forward discriminator.
        D = runner.ddp_models['discriminator']
        D_kwargs = runner.model_kwargs_train['discriminator']
        with ddp_sync(D, sync=sync):
            return D(images, labels, **D_kwargs)

    @staticmethod
    def compute_grad_penalty(images, scores):
        """Computes gradient penalty."""
        with conv2d_gradfix.no_weight_gradients():
            image_grad = torch.autograd.grad(
                outputs=[scores.sum()],
                inputs=[images],
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        grad_penalty = image_grad.square().sum((1, 2, 3))
        return grad_penalty

    def compute_pl_penalty(self, images, latents):
        """Computes perceptual path length penalty."""
        res_h, res_w = images.shape[2:4]
        pl_noise = torch.randn_like(images) / np.sqrt(res_h * res_w)
        with conv2d_gradfix.no_weight_gradients():
            code_grad = torch.autograd.grad(
                outputs=[(images * pl_noise).sum()],
                inputs=[latents],
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        pl_length = code_grad.square().sum(2).mean(1).sqrt()
        pl_mean = self.pl_mean.lerp(pl_length.mean(), self.pl_decay)
        self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_length - pl_mean).square()
        return pl_penalty

    @staticmethod
    def _labels_from_logits(logits):
        """Returns labels from logits.
        
        This function is useful when the classifier was trained on only one class,
        so it is necessary to round the logits and then apply one_hot to obtain the
        labels with shape (batch_size, 2).

        Args:
            logits (torch.Tensor): Logits from the classifier. Shape (batch_size, 1).
        
        Returns:
            torch.Tensor: Labels from the logits. Shape (batch_size, 2).

        """
        binary_output = torch.tensor(F.sigmoid(logits.squeeze()).round(), dtype=torch.int64)
        return F.one_hot(binary_output, num_classes=2)

    @staticmethod
    def run_LPIPS(runner, a, b, sync=True):
        """Runs LPIPS Model."""
        # Forward classifier.
        LPIPS = runner.ddp_models['lpips']
        LPIPS_kwargs = runner.model_kwargs_train['lpips']
        with ddp_sync(LPIPS, sync=sync):
            return LPIPS(a, b, **LPIPS_kwargs)

    def reconstruction_loss(self, runner, data, sync=True):
        """Computes reconstruction loss."""

        real_img = data['image']
        real_encoded = self.run_E(runner, real_img, sync=True)
        real_logits = self.run_C(runner, real_img, sync=True)
        real_labels = self._labels_from_logits(real_logits)

        fake_results_from_encoded = self.run_G_from_latents_w(
            runner,
            real_encoded,
            real_labels,
            sync=sync,
        )
        fake_img = fake_results_from_encoded['image']
        fake_logits = self.run_C(runner, fake_img, sync=True)

        # KL Divergence between real and fake logits.
        kl_div = F.kl_div(
            F.log_softmax(real_logits, dim=1),
            F.softmax(fake_logits, dim=1),
            reduction='batchmean',
        )

        fake_encoded = self.run_E(runner, fake_img, sync=sync)

        # Reconstruction loss
        reconstruct_loss_x = F.l1_loss(fake_img, real_img)  # Shape (,)
        reconstruct_loss_latent = F.l1_loss(fake_encoded, real_encoded)  # Shape (,)
        reconstruct_loss_lpips = self.run_LPIPS(
            runner,
            fake_img / fake_img.max(),
            real_img / real_img.max(),
            sync=sync,
        ).flatten()  # Shape (1,). Flatten gets rid of extra dimensions.

        reconstruct_loss = (
            reconstruct_loss_x
            + reconstruct_loss_latent
            + reconstruct_loss_lpips
        )

        runner.running_stats.update({'Loss/Recon. Image L1': reconstruct_loss_x})
        runner.running_stats.update({'Loss/Recon. Class Logits KL': kl_div})
        runner.running_stats.update({'Loss/Recon. Latent z L1': reconstruct_loss_latent})
        runner.running_stats.update({'Loss/Recon. Image LPIPS': reconstruct_loss_lpips})

        return reconstruct_loss + kl_div  # Shape (1,)

    def g_loss(self, runner, _data, sync=True):
        """Computes loss for generator."""
        # TODO: Programar opción para que se condicione en w y no en z.
        
        images_real = _data['image']

        images_latent = self.run_E(runner, images_real, sync=True)
        images_logits = self.run_C(runner, images_real, sync=True)
        images_labels = self._labels_from_logits(images_logits)

        fake_results_from_encoded = self.run_G_from_latents_w(
            runner,
            images_latent,
            images_labels,
            sync=sync,
        )

        fake_results_from_noise = self.run_G(runner, sync=sync)
        fake_scores_from_encoded = self.run_D(
            runner,
            images=fake_results_from_encoded['image'],
            labels=fake_results_from_encoded['label'],
            sync=False,
        )['score']
        fake_scores_from_noise = self.run_D(
            runner,
            images=fake_results_from_noise['image'],
            labels=fake_results_from_noise['label'],
            sync=False,
        )['score']
        fake_scores = torch.cat(
            [
                fake_scores_from_encoded,
                fake_scores_from_noise,
            ],
            dim=0,
        )
        g_loss = F.softplus(-fake_scores)
        runner.running_stats.update({'Loss/G': g_loss})

        return g_loss.mean()

    def g_reg(self, runner, _data, sync=True):
        """Computes the regularization loss for generator."""
        if runner.iter % self.pl_interval != 1 or self.pl_weight == 0.0:
            return None

        batch_size = max(runner.batch_size // self.pl_batch_shrink, 1)
        fake_results = self.run_G(runner,
                                  batch_size=batch_size,
                                  sync=sync,
                                  requires_grad=True)
        pl_penalty = self.compute_pl_penalty(images=fake_results['image'],
                                             latents=fake_results['wp'])
        runner.running_stats.update({'Loss/Path Length Penalty': pl_penalty})
        pl_penalty = pl_penalty * self.pl_weight * self.pl_interval

        return (fake_results['image'][:, 0, 0, 0] * 0 + pl_penalty).mean()

    def d_fake_loss(self, runner, _data, sync=True):
        """Computes discriminator loss on generated images."""
        fake_results = self.run_G(runner, sync=False)
        fake_scores = self.run_D(runner,
                                 images=fake_results['image'],
                                 labels=fake_results['label'],
                                 sync=sync)['score']
        d_fake_loss = F.softplus(fake_scores)
        runner.running_stats.update({'Loss/D Fake': d_fake_loss})

        return d_fake_loss.mean()

    def d_real_loss(self, runner, data, sync=True):
        """Computes discriminator loss on real images."""
        real_images = data['image'].detach()
        real_labels = data.get('label', None)
        real_scores = self.run_D(runner,
                                 images=real_images,
                                 labels=real_labels,
                                 sync=sync)['score']
        d_real_loss = F.softplus(-real_scores)
        runner.running_stats.update({'Loss/D Real': d_real_loss})

        # Adjust the augmentation strength if needed.
        if hasattr(runner.augment, 'prob_tracker'):
            runner.augment.prob_tracker.update(real_scores.sign())

        return d_real_loss.mean()

    def d_reg(self, runner, data, sync=True):
        """Computes the regularization loss for discriminator."""
        if runner.iter % self.r1_interval != 1 or self.r1_gamma == 0.0:
            return None

        real_images = data['image'].detach().requires_grad_(True)
        real_labels = data.get('label', None)
        real_scores = self.run_D(runner,
                                 images=real_images,
                                 labels=real_labels,
                                 sync=sync)['score']
        r1_penalty = self.compute_grad_penalty(images=real_images,
                                               scores=real_scores)
        runner.running_stats.update({'Loss/Real Gradient Penalty': r1_penalty})
        r1_penalty = r1_penalty * (self.r1_gamma * 0.5) * self.r1_interval

        # Adjust the augmentation strength if needed.
        if hasattr(runner.augment, 'prob_tracker'):
            runner.augment.prob_tracker.update(real_scores.sign())

        return (real_scores * 0 + r1_penalty).mean()
