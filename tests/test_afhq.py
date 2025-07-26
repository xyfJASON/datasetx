import unittest

import torch
import torchvision.transforms as T
from torch import Tensor

from datasetx import AFHQ


class TestAFHQ(unittest.TestCase):

    root = '~/data/AFHQ'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing AFHQ dataset...' + '\033[0m')

    def check_sample(self, sample: dict):
        # check sample, {image, label}
        self.assertIsInstance(sample, dict)
        self.assertEqual(sample.keys(), {'image', 'label'})
        image, label = sample['image'], sample['label']
        # check image, size 512x512
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, 512, 512))
        self.assertEqual(image.dtype, torch.float32)
        # check label, [0, 3)
        self.assertIsInstance(label, int)
        self.assertGreaterEqual(label, 0)
        self.assertLess(label, 3)

    def test_train_split(self):
        train_set = AFHQ(self.root, split='train')
        self.assertEqual(len(train_set), 14336)
        self.check_sample(train_set[0])

    def test_test_split(self):
        test_set = AFHQ(self.root, split='test')
        self.assertEqual(len(test_set), 1467)
        self.check_sample(test_set[0])

    def test_transform_fn(self):
        train_set = AFHQ(
            root=self.root,
            split='train',
            transform_fn=AFHQTransform(size=256),
        )
        # check sample, {image, label}
        sample = train_set[0]
        self.assertIsInstance(sample, dict)
        self.assertEqual(sample.keys(), {'image', 'label'})
        image, label = sample['image'], sample['label']
        # check image, size 256x256
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, 256, 256))
        self.assertEqual(image.dtype, torch.float32)
        # check label, [0, 3)
        self.assertIsInstance(label, int)
        self.assertGreaterEqual(label, 0)
        self.assertLess(label, 3)


class AFHQTransform:
    def __init__(self, size: int = 256):
        self.image_transform = T.Compose([
            T.Resize((size, size), antialias=True),
            T.RandomHorizontalFlip(),
            T.Normalize(0.5, 0.5),
        ])

    def __call__(self, sample: dict):
        sample['image'] = self.image_transform(sample['image'])
        return sample
