# python3.7
"""Configuration for training Faster R-CNN."""

from .base_config import BaseConfig

__all__ = ['FasterRCNNConfig']

RUNNER = 'FasterRCNNRunner'
DATASET = 'ObjectDetectionDataset'
FASTER_RCNN = 'FasterRCNN'
LOSS = 'FasterRCNNLoss'


class FasterRCNNConfig(BaseConfig):
    """Defines the configuration for training Faster R-CNN."""

    name = 'faster_rcnn'
    hint = 'Train a Faster R-CNN model.'
    # TODO: Search for recommended hiperparameters
    info = '''
To train a Faster R-CNN model, the recommended settings are as follows:

\b
- pretrained: False (for ImaGenome Locations dataset)
- progress: True (Really not sure if it makes difference)
- num_classes: 36 (for ImaGenome Locations dataset)
- pretrained_backbone: False (for ImaGenome Locations dataset)
- trainable_backbone_layers: False (for ImaGenome Locations dataset)
- batch_size: 6 (for ImaGenome Locations dataset, 1 GPU 10GB)
- val_batch_size: (Don't know yet)
'''

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.config.runner_type = RUNNER

    @classmethod
    def get_options(cls):
        options = super().get_options()

        options['Data transformation settings'].extend([
            cls.command_option(
                '--resolution', type=cls.int_type, default=256,
                help='Resolution of the training images.'),
            cls.command_option(
                '--image_channels', type=cls.int_type, default=1,
                help='Number of channels of the training images.'),
            cls.command_option(
                '--min_val', type=cls.float_type, default=-1.0,
                help='Minimum pixel value of the training images.'),
            cls.command_option(
                '--max_val', type=cls.float_type, default=1.0,
                help='Maximum pixel value of the training images.')
        ])

        options['Training settings'].extend([
            cls.command_option(
                '--lr', type=cls.float_type, default=0.002,
                help='The learning rate of the model.'),
            cls.command_option(
                '--beta_1', type=cls.float_type, default=0.0,
                help='The Adam hyper-parameter `beta_1`.'),
            cls.command_option(
                '--beta_2', type=cls.float_type, default=0.99,
                help='The Adam hyper-parameter `beta_2`.')
        ])

        options['Network settings'].extend([
            cls.command_option(
                '--pretrained', type=cls.bool_type, default=False,
                help=''),
            cls.command_option(
                '--progress', type=cls.bool_type, default=True,
                help=''),
            cls.command_option(
                '--num_classes', type=cls.int_type, default=36,
                help=''),
            cls.command_option(
                '--pretrained_backbone', type=cls.bool_type, default=False,
                help=''),
            cls.command_option(
                '--trainable_backbone_layers', type=cls.int_type, default=3,
                help='')
        ])

        return options

    @classmethod
    def get_recommended_options(cls):
        recommended_opts = super().get_recommended_options()
        recommended_opts.extend([
            'resolution', 'latent_dim', 'label_dim', 'lr',
        ])
        return recommended_opts

    def parse_options(self):
        super().parse_options()

        resolution = self.args.pop('resolution')
        image_channels = self.args.pop('image_channels')
        min_val = self.args.pop('min_val')
        max_val = self.args.pop('max_val')

        # Parse data transformation settings.
        data_transform_kwargs = dict(
            image_size=resolution,
            image_channels=image_channels,
            min_val=min_val,
            max_val=max_val
        )
        self.config.data.train.dataset_type = DATASET
        self.config.data.train.transform_kwargs = data_transform_kwargs
        self.config.data.val.dataset_type = DATASET
        self.config.data.val.transform_kwargs = data_transform_kwargs

        # Parse network settings and training settings.
        lr = self.args.pop('lr')
        beta_1 = self.args.pop('beta_1')
        beta_2 = self.args.pop('beta_2')

        self.config.models.update(
            frcnn=dict(
                model=dict(model_type=FASTER_RCNN,
                           pretrained=self.args.pop('pretrained'),
                           progress=self.args.pop('progress'),
                           num_classes=self.args.pop('num_classes'),
                           pretrained_backbone=self.args.pop('pretrained_backbone'),
                           trainable_backbone_layers=self.args.pop('trainable_backbone_layers'),
                ),
                lr=dict(lr_type='FIXED'),
                opt=dict(
                    opt_type='Adam',
                    base_lr=lr,
                    betas=(beta_1, beta_2),
                ),
                kwargs_train=dict(),
                kwargs_val=dict(),
            ),
        )

        self.config.loss.update(
            loss_type=LOSS,
            loss_kwargs=dict(),
        )

        self.config.metrics.update(
            FasterRCNNLosses=dict(
                init_kwargs=dict(name='frnn_losses'),
                eval_kwargs=dict(
                    frcnn=dict(),
                ),
                interval=None,
                first_iter=None,
                save_best=True
            )
        )
