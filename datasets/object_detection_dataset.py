# python3.7
"""Contains the class of object detection dataset.

`ObjectDetectionDataset` is commonly used as the dataset that provides images with labels
and bounding boxes. Concretely, each data sample (or say item) consists of an image and
its corresponding labels and bounding boxes

As an example, this dataset should be useful to train a faster rcnn as in
https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
"""

import numpy as np

from .base_dataset import BaseDataset

__all__ = ['ObjectDetectionDataset']


class ObjectDetectionDataset(BaseDataset):
    """Defines the Object Detection dataset class.

    NOTE: Each image can be grouped with a simple label, like 0-9 for CIFAR10,
    or 0-999 for ImageNet. The returned item format is

    {
        'index': int,
        'raw_image': np.ndarray,
        'image': np.ndarray,
        'labels': List[int],  (one image will have many objects)
        'bboxes': List[Tuple[int]]  (each tuple is a bbox with 4 numbers)
    }

    Available transformation kwargs:

    - image_size: Final image size produced by the dataset. (required)
    - image_channels (default: 3)
    - min_val (default: -1.0)
    - max_val (default: 1.0)
    """

    def __init__(self,
                 root_dir,
                 file_format='zip',
                 annotation_path=None,
                 annotation_meta=None,
                 max_samples=-1,
                 transform_kwargs=None,
                 num_classes=None):
        """Initializes the dataset.

        Args:
            num_classes: Number of classes. If not provided, the dataset will
                parse all labels to get the maximum value. This field can also
                be provided as a number larger than the actual number of
                classes. For example, sometimes, we may want to leave an
                additional class for an auxiliary task. (default: None)
        """
        super().__init__(root_dir=root_dir,
                         file_format=file_format,
                         annotation_path=annotation_path,
                         annotation_meta=annotation_meta,
                         annotation_format='json',
                         max_samples=max_samples,
                         mirror=False,
                         transform_kwargs=transform_kwargs)

        self.dataset_classes = 0  # Number of classes contained in the dataset.
        self.num_classes = 0  # Actual number of classes provided by the loader.

        item_sample = self.items[0]
        assert isinstance(item_sample, dict)
        for k in ['img_path', 'labels', 'bboxes']:
            assert k in list(item_sample.keys())

        self.dataset_classes = max([
            max(item['labels']) for item in self.items
        ]) + 1

        if num_classes is None:
            self.num_classes = self.dataset_classes
        else:
            self.num_classes = int(num_classes)
        assert self.num_classes > 0

    def get_raw_data(self, idx):

        image_path = self.items[idx]['img_path']
        labels = self.items[idx]['labels']
        bboxes = self.items[idx]['bboxes']

        # Load image to buffer.
        buffer = np.frombuffer(self.fetch_file(image_path), dtype=np.uint8)
        idx = np.array(idx)

        return [idx, buffer, labels, bboxes]

    @property
    def num_raw_outputs(self):
        return 4  # [idx, buffer, labels, bboxes]

    def parse_transform_config(self):
        image_size = self.transform_kwargs.get('image_size')
        image_channels = self.transform_kwargs.setdefault('image_channels', 3)
        min_val = self.transform_kwargs.setdefault('min_val', -1.0)
        max_val = self.transform_kwargs.setdefault('max_val', 1.0)
        self.transform_config = dict(
            decode=dict(transform_type='Decode', image_channels=image_channels,
                        return_square=True, center_crop=True),
            resize=dict(transform_type='Resize', image_size=image_size),
            normalize=dict(transform_type='Normalize',
                           min_val=min_val, max_val=max_val)
        )

    def transform(self, raw_data, use_dali=False):
        idx, buffer, labels, bboxes = raw_data

        raw_image = self.transforms['decode'](buffer, use_dali=use_dali)
        raw_image = self.transforms['resize'](raw_image, use_dali=use_dali)
        image = self.transforms['normalize'](raw_image, use_dali=use_dali)

        return [idx, raw_image, image, labels, bboxes]

    def batch_to_device(self, batch):
        images, targets = batch['images'], batch['targets']

        images = list(image.cuda() for image in images)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        return images, targets

    @property
    def output_keys(self):
        return ['index', 'raw_image', 'image', 'labels', 'bboxes']

    def info(self):
        dataset_info = super().info()
        dataset_info['Dataset classes'] = self.dataset_classes
        dataset_info['Use label'] = self.use_label
        dataset_info['Num classes for training'] = self.num_classes
        return dataset_info
