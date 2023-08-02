# python3.7
"""Contains the class of directory reader.

This reader can summarize file list or fetch bytes of files inside a directory.
"""

import os.path

from .base_reader import BaseReader

__all__ = ['JPGDirectoryReader', 'PNGDirectoryReader', 'DirectoryReader']


class DirectoryReader(BaseReader):
    """Defines a class to load directory."""

    @staticmethod
    def open(path):
        assert os.path.isdir(path), f'Directory `{path}` is invalid!'
        return path

    @staticmethod
    def close(path):
        _ = path  # Dummy function.
    
    @staticmethod
    def _valid_path_criterion(path):
        return True

    @staticmethod
    def open_anno_file(path, anno_filename=None):
        path = DirectoryReader.open(path)
        if not anno_filename:
            return None
        anno_path = os.path.join(path, anno_filename)
        if not os.path.isfile(anno_path):
            return None
        # File will be closed after parsed in dataset.
        return open(anno_path, 'r')

    @classmethod
    def _get_file_list(cls, path):
        path = DirectoryReader.open(path)
        paths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                if cls._valid_path_criterion(file_path):
                    paths.append(file_path[len(path) + 1:])
        return paths

    @staticmethod
    def fetch_file(path, filename):
        path = DirectoryReader.open(path)
        with open(os.path.join(path, filename), 'rb') as f:
            file_bytes = f.read()
        return file_bytes


class CheXpertFrontalReader(DirectoryReader):

    @staticmethod
    def _valid_path_criterion(path):
        return path.endswith(".jpg") and "frontal" in path


class PNGDirectoryReader(DirectoryReader):

    @staticmethod
    def _valid_path_criterion(path):
        return path.endswith(".png")


class JPGDirectoryReader(DirectoryReader):

    @staticmethod
    def _valid_path_criterion(path):
        return path.endswith(".jpg")
