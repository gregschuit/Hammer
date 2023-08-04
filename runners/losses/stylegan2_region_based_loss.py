# python3.7
"""Defines loss functions for StyleGAN2 training with region-based penalty."""

import torch
import torch.nn.functional as F

from utils.dist_utils import ddp_sync
from .stylegan2_loss import StyleGAN2Loss

# TODO: This should be other model file
from ultralytics import YOLO
from torchvision.transforms.functional import gaussian_blur

__all__ = ['StyleGAN2RegionBasedLoss']

YOLO_WEIGHTS_PATH = '/home/gregschuit/projects/cxr-object-detection/runs/detect/train/weights/best.pt'


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


def get_mask(
    img,
    box,
    device,
    use_soft_box=False,
    soft_box_margin=40,
    soft_box_kernel_size=51,
    soft_box_sigma=30,
):
    """Constructs a mask for the given image and bounding box.
    
    Args:
        img (torch.Tensor): Image tensor of shape (C, H, W).

    """
    mask = torch.ones(img.shape)
    mask = mask.to(device)
    x1, y1, x2, y2 = map(int, box)
    mask[:, y1:y2, x1:x2] = 0

    if use_soft_box:
        margin = soft_box_margin
        mask[:, y1 - margin : y2 + margin, x1 - margin : x2 + margin] = 0
        mask = gaussian_blur(
            mask,
            kernel_size=soft_box_kernel_size,
            sigma=soft_box_sigma,
        )

    return mask


def strict_divide(number, divisor, msg=None):
    """Divides `number` by `divisor` and asserts that the result is an integer."""
    msg = msg or f'{number} is not divisible by {divisor}'
    assert number % divisor == 0, msg
    return number // divisor


class StyleGAN2RegionBasedLoss(StyleGAN2Loss):
    """Contains the class to compute losses for training StyleGAN2.

    Basically, this class contains the computation of adversarial loss for both
    generator and discriminator, perceptual path length regularization for
    generator, and gradient penalty as the regularization for discriminator.
    """

    def __init__(self, runner, d_loss_kwargs=None, g_loss_kwargs=None):
        """Initializes with models and arguments for computing losses."""

        super().__init__(runner, d_loss_kwargs, g_loss_kwargs)

        self.region_based_args = g_loss_kwargs.get('region_based', dict())
        self.penalty_weight = self.region_based_args.get('penalty_weight', 1.0)
        self.region_based_args['use_soft_box'] = self.region_based_args.get(
            'use_soft_box', False
        )
        self.region_based_args['soft_box_margin'] = self.region_based_args.get(
            'soft_box_margin', 40
        )
        self.region_based_args[
            'soft_box_kernel_size'
        ] = self.region_based_args.get('soft_box_kernel_size', 51)
        self.region_based_args[
            'soft_box_sigma'
        ] = self.region_based_args.get('soft_box_sigma', 30)

        self.yolo = YOLO(YOLO_WEIGHTS_PATH, task='detect')
        self.yolo.to(runner.device)

        runner.running_stats.add('Loss/OutOfBoxPenalty',
                                 log_name='oob_penalty',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')

    @staticmethod
    def run_G(runner, batch_size=None, sync=True, requires_grad=False):
        """Forwards generator.

        NOTE: The flag `requires_grad` sets whether to compute the gradient for
            latent z. When computing the `pl_penalty` with part of the generator
            frozen (e.g., mapping network), this flag should be set to `True` to
            retain the computation graph.
        """
        batch_size = batch_size or runner.batch_size
        batch_size = strict_divide(
            batch_size, 2, 'Batch size must be even when using region-based loss.'
        )

        latent_dim = runner.models['generator'].latent_dim
        label_dim = runner.models['generator'].label_dim

        if label_dim and (label_dim != 2):
            raise ValueError(
                'Region-based loss only supports label_dim == 2 (binary setting).'
            )

        # Initiate half negative and half positive labels.
        negatives = torch.zeros(batch_size, device=runner.device, dtype=torch.int64)
        positives = torch.ones(batch_size, device=runner.device, dtype=torch.int64)
        labels = torch.cat([negatives, positives], dim=0)
        labels = F.one_hot(labels, num_classes=2)

        # Initiate duplicate latent codes (the same for negative and positive samples).
        latents = torch.randn((batch_size, *latent_dim),
                              device=runner.device,
                              requires_grad=requires_grad)
        latents = torch.cat([latents, latents], dim=0)

        # Forward generator.
        G = runner.ddp_models['generator']
        G_kwargs = runner.model_kwargs_train['generator']
        with ddp_sync(G, sync=sync):
            return G(latents, labels, **G_kwargs)

    def _region_based_penalty(self, runner, fake_results, sync=True):
        # Post process batch (necessary for YOLO).
        images = postprocess_batch(fake_results['image'])
        negative_images = images[0:runner.batch_size // 2]
        positive_images = images[runner.batch_size // 2:]

        # Predict bounding boxes over negative images.
        with torch.no_grad():
            results = self.yolo.predict(
                negative_images.repeat(1, 3, 1, 1),
                verbose=False,
            )

        # The following part is anomaly-specific. In this case, we hard-code the
        # class index for the cardiac box.
        # TODO: Generalize this part to other anomalies.
        cardiac_boxes_idxs = [find_cardiac_box(result) for result in results]
        cardiac_boxes = [
            get_box(result, idx) for result, idx in zip(results, cardiac_boxes_idxs)
        ]

        masks = [
            get_mask(
                img,
                box.cpu().tolist(),
                device=runner.device,
                use_soft_box=self.region_based_args['use_soft_box'],
                soft_box_margin=self.region_based_args['soft_box_margin'],
                soft_box_kernel_size=self.region_based_args[
                    'soft_box_kernel_size'
                ],
                soft_box_sigma=self.region_based_args[
                    'soft_box_sigma'
                ],
            ) for img, box in zip(images, cardiac_boxes)
        ]
        mask = torch.cat(masks, axis=0).unsqueeze(1)

        # Substract images
        sqr_err = (negative_images - positive_images) ** 2
        masked_sqr_err = sqr_err * mask
        masked_mse = torch.sum(masked_sqr_err) / torch.sum(mask)

        runner.running_stats.update({'Loss/OutOfBoxPenalty': masked_mse})

        return masked_mse

    def g_loss(self, runner, _data, sync=True):
        """Computes loss for generator."""

        fake_results = self.run_G(runner, sync=sync)

        region_based_penalty = self._region_based_penalty(
            runner, fake_results, sync=sync
        )

        fake_scores = self.run_D(runner,
                                 images=fake_results['image'],
                                 labels=fake_results['label'],
                                 sync=False)['score']
        g_loss = F.softplus(-fake_scores)
        runner.running_stats.update({'Loss/G': g_loss})

        return g_loss.mean() + self.penalty_weight * region_based_penalty
