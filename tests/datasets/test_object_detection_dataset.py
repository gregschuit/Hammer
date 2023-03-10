import numpy as np
import pytest

from datasets.object_detection_dataset import ObjectDetectionDataset

from dotenv import dotenv_values

env = dotenv_values()


class TestObjectDetectionDataset:

    @pytest.fixture
    def set_up(self):
        self.ds = ObjectDetectionDataset(
            root_dir=env['MIMIC_CXR_JPG_DIR_TRAIN'],
            file_format='dir',
            annotation_path=env['ANNOTATIONS_LOCATIONS_PATH'],
            max_samples=-1,
            transform_kwargs={
                'image_size': 256,
                'image_channels': 1,
                'min_val': -1,
                'max_val': 1,
            }
        )

    @staticmethod
    def _check_bboxes_structure(bboxes):
        assert type(bboxes) == list
        assert type(bboxes[0]) == list
        assert len(bboxes[0]) == 4

    @staticmethod
    def _check_labels_structure(labels):
        assert type(labels) == list
        assert type(labels[0]) == int

    def test_raw_data(self, set_up):
        idx = 0
        result = self.ds.get_raw_data(idx)

        assert len(result) == self.ds.num_raw_outputs
        _idx, buffer, labels, bboxes = result

        assert _idx == np.array(idx)

        assert type(buffer) == np.ndarray
        assert len(buffer.shape) == 1

        self._check_labels_structure(labels)
        self._check_bboxes_structure(bboxes)

    def test_transform(self, set_up):
        idx = 0
        raw_data = self.ds.get_raw_data(idx)
        result = self.ds.transform(raw_data)

        idx, raw_image, image, labels, bboxes = result
        assert raw_image.shape == (256, 256, 1)
        assert image.shape == (1, 256, 256)
        self._check_labels_structure(labels)
        self._check_bboxes_structure(bboxes)
