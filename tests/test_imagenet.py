import unittest

import torch
import torchvision.transforms as T
from torch import Tensor

from datasetx import ImageNet


class TestImageNet(unittest.TestCase):

    root = '~/data/ImageNet/ILSVRC2012/Images'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing ImageNet dataset...' + '\033[0m')

    def check_sample(self, sample: dict, size: int = None, label_is_none: bool = False):
        # check sample, {image, label}
        self.assertIsInstance(sample, dict)
        self.assertEqual(sample.keys(), {'image', 'label'})
        image, label = sample['image'], sample['label']
        # check image
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.dtype, torch.float32)
        if size is not None:
            self.assertEqual(image.shape, (3, size, size))
        # check label, [0, 1000)
        if not label_is_none:
            self.assertIsInstance(label, int)
            self.assertGreaterEqual(label, 0)
            self.assertLess(label, 1000)
        else:
            self.assertIsNone(label)

    def test_train_split(self):
        train_set = ImageNet(self.root, split='train')
        self.assertEqual(len(train_set), 1281167)
        self.check_sample(train_set[0])

    def test_valid_split(self):
        valid_set = ImageNet(self.root, split='valid')
        self.assertEqual(len(valid_set), 50000)
        self.check_sample(valid_set[0])

    def test_test_split(self):
        test_set = ImageNet(self.root, split='test')
        self.assertEqual(len(test_set), 100000)
        self.check_sample(test_set[0], label_is_none=True)

    def test_transform_fn(self):
        train_set = ImageNet(
            root=self.root,
            split='train',
            transform_fn=ImageNetTransform(size=256),
        )
        self.check_sample(train_set[0], size=256)


class ImageNetTransform:
    def __init__(self, size: int = 256):
        self.image_transform = T.Compose([
            T.Resize(size, antialias=True),
            T.CenterCrop(size),
            T.RandomHorizontalFlip(),
            T.Normalize(0.5, 0.5),
        ])

    def __call__(self, sample: dict):
        sample['image'] = self.image_transform(sample['image'])
        return sample
