# python3.7
"""Defines loss functions for StyleGAN2 training."""

import numpy as np

import torch
import torch.nn.functional as F

from third_party.stylegan2_official_ops import conv2d_gradfix
from utils.dist_utils import ddp_sync
from .base_loss import BaseLoss

__all__ = ['StyleGAN2Loss']

from ultralytics import YOLO
from torchvision.transforms.functional import gaussian_blur, to_pil_image, pil_to_tensor


REGION_BASED = True
SOTF_BOX = False
PENALTY_W = 10


def postprocess_batch(generated_images):
    _min, _max = -1, 1
    generated_images = torch.clamp(generated_images, _min, _max)
    generated_images = (generated_images - _min) / (_max - _min)
    return generated_images


def find_cardiac_box(result):
    try:
        return (result.boxes.cls == 2).nonzero()[0][0].item()
    except IndexError:
        return None


def get_box(result, idx):
    default_box = torch.tensor([95, 121, 196, 194])  # Avg box from training set
    if idx is None:
        return default_box
    return result.boxes.xyxy[idx]


def get_mask(img, box, device):
    mask = torch.ones_like(pil_to_tensor(img))
    mask = mask.to(device)
    x1, y1, x2, y2 = map(int, box)
    mask[:, y1:y2, x1:x2] = 0

    if SOTF_BOX:
        margin = 40
        mask[:, y1 - margin : y2 + margin, x1 - margin : x2 + margin] = 0
        mask = gaussian_blur(mask, 51, sigma=30)

    return mask


class StyleGAN2Loss(BaseLoss):
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

        self.yolo = YOLO('/home/gregschuit/projects/cxr-object-detection/runs/detect/train/weights/best.pt', task='detect')
        self.yolo.to(runner.device)
        runner.running_stats.add('Loss/OutOfBoxPenalty',
                                 log_name='oob_penalty',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')

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

    @staticmethod
    def run_G(runner, batch_size=None, sync=True, requires_grad=False):
        """Forwards generator.

        NOTE: The flag `requires_grad` sets whether to compute the gradient for
            latent z. When computing the `pl_penalty` with part of the generator
            frozen (e.g., mapping network), this flag should be set to `True` to
            retain the computation graph.
        """
        # Prepare latent codes and labels.
        batch_size = batch_size or runner.batch_size

        if REGION_BASED:
            batch_size /= 2
            if batch_size % 1 != 0:
                raise ValueError('`batch_size` must be even when using region_based loss.')
            batch_size = int(batch_size)

        latent_dim = runner.models['generator'].latent_dim
        label_dim = runner.models['generator'].label_dim
        latents = torch.randn((batch_size, *latent_dim),
                              device=runner.device,
                              requires_grad=requires_grad)

        if not REGION_BASED:
            labels = None
            if label_dim > 0:
                rnd_labels = torch.randint(
                    0, label_dim, (batch_size,), device=runner.device)
                labels = F.one_hot(rnd_labels, num_classes=label_dim)
        else:
            positives = torch.ones(batch_size, device=runner.device, dtype=torch.int64)
            negatives = torch.zeros(batch_size, device=runner.device, dtype=torch.int64)
            labels = torch.cat([negatives, positives], dim=0)
            labels = F.one_hot(labels, num_classes=2)

        if REGION_BASED:
            latents = torch.cat([latents, latents], dim=0)

        # Forward generator.
        G = runner.ddp_models['generator']
        G_kwargs = runner.model_kwargs_train['generator']
        with ddp_sync(G, sync=sync):
            return G(latents, labels, **G_kwargs)

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

    def g_loss(self, runner, _data, sync=True):
        """Computes loss for generator."""

        # Primero debería implementarlo sin region_based loss.
        # Entonces:
        # 1. Parametrizar el uso de region_based loss.
        # 2. Programar el uso del encoder para embeber las imágenes reales.

        # Opción 1:
        # Tomar la mitad del batch, y usar imagenes reales.
        # Pasar las imágenes reales por el encoder y por el clasificador para obtener w.
        # Pasar z por las FF layers para obtener w.
        # Juntar ambos w y seguir con el flujo normal.

        # Opción 2:
        # Tomar la mitad del batch, y usar imagenees reales.
        # Usar solo imágenes reales positivas.
        # Tomar también sus bboxes.
        # Más adelante, cuando las imágenes se generen, usar las bboxes para
        # generar las máscaras y usarlas para calcular la loss.

        # Problema: El D se podría sesgar si todas las reales son positivas.

        # Para la opción 1.
        # En concreto:
        # 1. Obtener imágenes reales. Esta viene en la variable _data.
        # 2. Cargar el encoder. Por ahora puede ser una variable global?
        # 3. Calcular cuántas imágenes equivalen a la mitad del batch (o a un cuarto).
        # 4. Pasar las imágenes por el encoder.
        # 5. Probar con z o con w?
        # 6. Si elijo w, tengo que editar la condicionalidad para que siempre condicione
        #    con w, y no con z.

        fake_results = self.run_G(runner, sync=sync)

        # Post process batch
        images = postprocess_batch(fake_results['image'])
        negative_images = images[0:runner.batch_size // 2]
        positive_images = images[runner.batch_size // 2:]

        with torch.no_grad():
            pil_imgs = [to_pil_image(img) for img in negative_images]
            results = self.yolo.predict(pil_imgs, verbose=False)

        cardiac_boxes_idxs = [find_cardiac_box(result) for result in results]
        cardiac_boxes = [
            get_box(result, idx) for result, idx in zip(results, cardiac_boxes_idxs)
        ]

        masks = [
            get_mask(
                img, box.cpu().tolist(),
                device=runner.device
            ) for img, box in zip(pil_imgs, cardiac_boxes)
        ]
        mask = torch.cat(masks, axis=0).unsqueeze(1)

        # Substract images
        sqr_err = (negative_images - positive_images) ** 2
        masked_sqr_err = sqr_err * mask
        masked_mse = torch.sum(masked_sqr_err) / torch.sum(mask)

        fake_scores = self.run_D(runner,
                                 images=fake_results['image'],
                                 labels=fake_results['label'],
                                 sync=False)['score']
        g_loss = F.softplus(-fake_scores)
        runner.running_stats.update({'Loss/G': g_loss})
        runner.running_stats.update({'Loss/OutOfBoxPenalty': masked_mse})

        return g_loss.mean() + PENALTY_W * masked_mse

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
