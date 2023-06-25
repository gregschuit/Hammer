# python3.7
"""Contains the base class for GAN-related metric computation.

Different from other deep models, evaluating a generative model (especially the
generator part) often requires to set up a collection of synthesized data beyond
the given validation set. For this purpose, it requires to sample a collection
of latent codes. To ensure the reproducibility as well as the evaluation
consistency during the training process, one may need to specify the latent code
collection and also save the collection used. Accordingly, this base class
handles the latent codes loading, splitting to replicas, and saving.
"""

import os.path
import time
import numpy as np

import torch
import torch.nn.functional as F

from .base_metric import BaseMetric

__all__ = ['FasterRCNNLossesMetric']


class FasterRCNNLossesMetric(BaseMetric):

    def __init__(self,
                 name='frcnn_losses',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1):
        """Initialization with latent codes loading, splitting, and saving.

        Args:
            seed: Seed used for sampling. This is essential to ensure the
                reproducibility. (default: 0)
        """
        super().__init__(name, work_dir, logger, tb_writer, batch_size)

    def evaluate(self, data_loader, faster_rcnn, faster_rcnn_kwargs):
        losses = self.get_losses(data_loader, faster_rcnn)
        if self.is_chief:
            mean_losses = losses.mean(axis=0)
            result = {
                'loss_box_reg_mean': mean_losses[0],
                'loss_classifier_mean': mean_losses[1],
                'loss_objectness_mean': mean_losses[2],
                'loss_rpn_box_reg_mean': mean_losses[3],
            }
        else:
            assert losses is None
            result = None
        self.sync()
        return result
    
    def get_losses(self, data_loader, faster_rcnn):
        real_num = len(data_loader.dataset)

        self.logger.info(f'Extracting inception features from real data '
                         f'{self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('Eval', total=real_num)

        all_results = []
        batch_size = data_loader.batch_size
        replica_num = self.get_replica_num(real_num)
        for batch_idx in range(len(data_loader)):
            if batch_idx * batch_size >= replica_num:
                # NOTE: Here, we always go through the entire dataset to make
                # sure the next evaluator can visit the data loader from the
                # beginning.
                _batch_data = next(data_loader)
                continue
            with torch.no_grad():
                batch_data = next(data_loader)
                images, targets = data_loader.dataset.batch_to_device(batch_data, batch_size)

                losses = faster_rcnn(images, targets)

                results = torch.tensor([
                    [
                        losses['loss_box_reg'].item(),
                        losses['loss_classifier'].item(),
                        losses['loss_objectness'].item(),
                        losses['loss_rpn_box_reg'].item(),
                    ]
                ])
                results = self.gather_batch_results(results)
                self.append_batch_results(results, all_results)
            self.logger.update_pbar(pbar_task, batch_size * self.world_size)
        self.logger.close_pbar()
        all_results = self.gather_all_results(all_results)[:real_num]

        if self.is_chief:
            assert all_results.shape[1] == 4  # 4 losses
        else:
            assert len(all_results) == 0
            all_results = None
        self.sync()
        return all_results

    def _is_better_than(self, metric_name, new, ref):
        """Lower losses are better."""
        return ref is None or new < ref

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        if not self.is_chief:
            assert result is None
            self.sync()
            return

        assert isinstance(result, dict)
        msg = f'Evaluating `{self.name}`: '

        for name, value in result.items():
            msg += f'{name} {value:.3f} '

        self.logger.info(msg)

        with open(os.path.join(self.work_dir, f'{self.name}.txt'), 'a+') as f:
            date = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'[{date}] {msg}\n')

        # Save to TensorBoard if needed.
        if self.tb_writer is not None:
            if tag is None:
                self.logger.warning('`Tag` is missing when writing data to '
                                    'TensorBoard, hence, the data may be mixed '
                                    'up!')
            self.tb_writer.add_scalar(
                f'Metrics/FRCNN Val Box Reg/',
                result['loss_box_reg_mean'],
                tag,
            )
            self.tb_writer.add_scalar(
                f'Metrics/FRCNN Val Classifier',
                result['loss_classifier_mean'],
                tag,
            )
            self.tb_writer.add_scalar(
                f'Metrics/FRCNN Val Objectness',
                result['loss_objectness_mean'],
                tag,
            )
            self.tb_writer.add_scalar(
                f'Metrics/FRCNN Val RPN Box',
                result['loss_rpn_box_reg_mean'],
                tag,
            )
            self.tb_writer.flush()
        self.sync()

    def info(self):
        metric_info = super().info()
        return metric_info
