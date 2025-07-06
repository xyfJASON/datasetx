import unittest
from PIL import Image

import torch
import torchvision.transforms as T

from image_datasets import ImageNet


class TestImageNet(unittest.TestCase):

    root = '~/data/ImageNet/ILSVRC2012/Images'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing ImageNet dataset...' + '\033[0m')

    def test_train_split(self):
        train_set = ImageNet(self.root, split='train')
        # check length
        self.assertEqual(len(train_set), 1281167)
        # check data, (image, label)
        data = train_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 2)
        image, label = data
        # check image
        self.assertIsInstance(image, Image.Image)
        # check label, [0, 1000)
        self.assertIsInstance(label, int)
        self.assertGreaterEqual(label, 0)
        self.assertLess(label, 1000)

    def test_valid_split(self):
        valid_set = ImageNet(self.root, split='valid')
        # check length
        self.assertEqual(len(valid_set), 50000)
        # check data, (image, label)
        data = valid_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 2)
        image, label = data
        # check image
        self.assertIsInstance(image, Image.Image)
        # check label, [0, 1000)
        self.assertIsInstance(label, int)
        self.assertGreaterEqual(label, 0)
        self.assertLess(label, 1000)

    def test_test_split(self):
        test_set = ImageNet(self.root, split='test')
        # check length
        self.assertEqual(len(test_set), 100000)
        # check data, (image, label)
        data = test_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 2)
        image, label = data
        # check image
        self.assertIsInstance(image, Image.Image)
        # check label, None
        self.assertIsNone(label)

    def test_train_split_with_transforms(self):
        transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        train_set = ImageNet(
            root=self.root,
            split='train',
            transforms=transforms,
        )
        # check data, (image, label)
        data = train_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 2)
        image, label = data
        # check image, tensor (3, 256, 256)
        self.assertEqual(image.shape, (3, 256, 256))
        self.assertEqual(image.dtype, torch.float32)
        # check label, [0, 1000)
        self.assertIsInstance(label, int)
        self.assertGreaterEqual(label, 0)
        self.assertLess(label, 1000)
