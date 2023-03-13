import numpy as np
import cv2
from datasets.object_detection_dataset import ObjectDetectionDataset

from dotenv import dotenv_values

env = dotenv_values()


class TestObjectDetectionDataset:

    def _build_ds(self, img_size=256, img_channels=1):
        self.img_size = img_size
        self.img_channels = img_channels
        self.ds = ObjectDetectionDataset(
            root_dir=env['MIMIC_CXR_JPG_DIR_TRAIN'],
            file_format='dir',
            annotation_path=env['ANNOTATIONS_LOCATIONS_PATH'],
            max_samples=-1,
            transform_kwargs={
                'image_size': self.img_size,
                'image_channels': self.img_channels,
                'min_val': -1,
                'max_val': 1,
            }
        )

    @staticmethod
    def _build_random_raw_data(img_size, img_channels):
        idx = np.array(np.random.randint(0, 100))
        img = np.random.randint(0, 255, size=(img_size, img_size, img_channels), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        n_objects_in_img = np.random.randint(1, 5)
        n_classes = 14
        labels = np.random.randint(0, n_classes, size=n_objects_in_img).tolist()
        bboxes = np.random.random(size=(n_objects_in_img, 4))
        bboxes[:, 2:4] = bboxes[:, 0:2] + bboxes[:, 2:4]
        bboxes *= img_size
        bboxes = [bbox.tolist() for bbox in bboxes]
        return [idx, buffer, labels, bboxes]

    @staticmethod
    def _check_bboxes_structure(bboxes):
        assert type(bboxes) == list
        assert type(bboxes[0]) == list
        assert len(bboxes[0]) == 4

    @staticmethod
    def _check_labels_structure(labels):
        assert type(labels) == list
        assert type(labels[0]) == int

    def test_raw_data(self):
        self._build_ds()

        idx = 0
        result = self.ds.get_raw_data(idx)

        assert len(result) == self.ds.num_raw_outputs
        _idx, buffer, labels, bboxes = result

        assert _idx == np.array(idx)

        assert type(buffer) == np.ndarray
        assert len(buffer.shape) == 1

        self._check_labels_structure(labels)
        self._check_bboxes_structure(bboxes)

    def _test_transform(
            self,
            original_img_size=256,
            original_img_channels=1,
            img_size=256,
            img_channels=1,
            expected_raw_image_shape=(256, 256, 1),
            expected_image_shape=(1, 256, 256),
        ):

        self._build_ds(img_size, img_channels)
        raw_data = self._build_random_raw_data(original_img_size, original_img_channels)

        result = self.ds.transform(raw_data)

        idx, raw_image, image, labels, bboxes = result
        assert raw_image.shape == expected_raw_image_shape
        assert image.shape == expected_image_shape
        self._check_labels_structure(labels)
        self._check_bboxes_structure(bboxes)

    def test_transform_equal_shape_and_channels(self):
        """It is expected that shapes will stay the same."""
        self._test_transform(
            original_img_size=256,
            original_img_channels=1,
            img_size=256,
            img_channels=1,
            expected_raw_image_shape=(256, 256, 1),
            expected_image_shape=(1, 256, 256)
        )

    def test_transform_other_shape(self):
        """It is expected that img_size will be changed."""
        self._test_transform(
            original_img_size=256,
            original_img_channels=1,
            img_size=128,
            img_channels=1,
            expected_raw_image_shape=(128, 128, 1),
            expected_image_shape=(1, 128, 128)
        )

    def test_transform_more_channels(self):
        """It is expected that more channels won't be added."""
        self._test_transform(
            original_img_size=256,
            original_img_channels=1,
            img_size=256,
            img_channels=3,
            expected_raw_image_shape=(256, 256, 1),
            expected_image_shape=(1, 256, 256)
        )

    def test_transform_less_channels(self):
        """It is expected that channels wont be changed."""
        self._test_transform(
            original_img_size=256,
            original_img_channels=3,
            img_size=256,
            img_channels=1,
            expected_raw_image_shape=(256, 256, 3),
            expected_image_shape=(3, 256, 256)
        )
