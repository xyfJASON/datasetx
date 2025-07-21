import unittest

import torch
import torchvision.transforms as T
from torch import Tensor

from image_datasets import FFHQ


class TestFFHQ(unittest.TestCase):

    root = '~/data/FFHQ'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing FFHQ dataset...' + '\033[0m')

    def check_sample(self, sample: dict, size: int):
        image = sample['image']
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, size, size))
        self.assertEqual(image.dtype, torch.float32)

    def test_train_split(self):
        train_set = FFHQ(self.root, split='train')
        self.assertEqual(len(train_set), 60000)
        self.check_sample(train_set[0], size=1024)

    def test_test_split(self):
        test_set = FFHQ(self.root, split='test')
        self.assertEqual(len(test_set), 10000)
        self.check_sample(test_set[0], size=1024)

    def test_all_split(self):
        all_set = FFHQ(self.root, split='all')
        self.assertEqual(len(all_set), 70000)
        self.check_sample(all_set[0], size=1024)

    def test_thumbnails_version(self):
        all_set = FFHQ(self.root, split='all', version='thumbnails128x128')
        self.assertEqual(len(all_set), 70000)
        self.check_sample(all_set[0], size=128)

    def test_transform_fn(self):
        train_set = FFHQ(
            root=self.root,
            split='train',
            transform_fn=FFHQTransform(size=512),
        )
        self.check_sample(train_set[0], size=512)


class FFHQTransform:
    def __init__(self, size: int = 512):
        self.image_transform = T.Compose([
            T.Resize((size, size), antialias=True),
            T.RandomHorizontalFlip(),
            T.Normalize(0.5, 0.5),
        ])

    def __call__(self, sample: dict):
        sample['image'] = self.image_transform(sample['image'])
        return sample
