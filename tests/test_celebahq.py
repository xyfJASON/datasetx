import unittest

import torch
import torchvision.transforms as T
from torch import Tensor

from datasetx import CelebAHQ


class TestCelebAHQ(unittest.TestCase):

    root = '~/data/CelebA-HQ'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing CelebA-HQ dataset...' + '\033[0m')

    def check_sample(self, sample: dict, size: int):
        image = sample['image']
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, size, size))
        self.assertEqual(image.dtype, torch.float32)

    def test_train_split(self):
        train_set = CelebAHQ(self.root, split='train')
        self.assertEqual(len(train_set), 24183)
        self.check_sample(train_set[0], size=1024)

    def test_valid_split(self):
        valid_set = CelebAHQ(self.root, split='valid')
        self.assertEqual(len(valid_set), 2993)
        self.check_sample(valid_set[0], size=1024)

    def test_test_split(self):
        test_set = CelebAHQ(self.root, split='test')
        self.assertEqual(len(test_set), 2824)
        self.check_sample(test_set[0], size=1024)

    def test_all_split(self):
        all_set = CelebAHQ(self.root, split='all')
        self.assertEqual(len(all_set), 30000)
        self.check_sample(all_set[0], size=1024)

    def test_transform_fn(self):
        train_set = CelebAHQ(
            root=self.root,
            split='train',
            transform_fn=CelebAHQTransform(size=256),
        )
        self.check_sample(train_set[0], size=256)


class CelebAHQTransform:
    def __init__(self, size: int = 256):
        self.image_transform = T.Compose([
            T.Resize((size, size), antialias=True),
            T.RandomHorizontalFlip(),
            T.Normalize(0.5, 0.5),
        ])

    def __call__(self, sample: dict):
        sample['image'] = self.image_transform(sample['image'])
        return sample
