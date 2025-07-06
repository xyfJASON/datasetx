import unittest
from PIL import Image

import torch
import torchvision.transforms as T

from image_datasets import AFHQ


class TestAFHQ(unittest.TestCase):

    root = '~/data/AFHQ'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing AFHQ dataset...' + '\033[0m')

    def test_train_split(self):
        train_set = AFHQ(self.root, split='train')
        # check length
        self.assertEqual(len(train_set), 14336)
        # check data, (image, label)
        data = train_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 2)
        image, label = data
        # check image, size 512x512
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (512, 512))
        # check label, [0, 3)
        self.assertIsInstance(label, int)
        self.assertGreaterEqual(label, 0)
        self.assertLess(label, 3)

    def test_test_split(self):
        test_set = AFHQ(self.root, split='test')
        # check length
        self.assertEqual(len(test_set), 1467)
        # check data, (image, label)
        data = test_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 2)
        image, label = data
        # check image, size 512x512
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (512, 512))
        # check label, [0, 3)
        self.assertIsInstance(label, int)
        self.assertGreaterEqual(label, 0)
        self.assertLess(label, 3)

    def test_train_split_with_transforms(self):
        transforms = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        train_set = AFHQ(
            root=self.root,
            split='train',
            transforms=transforms,
        )
        # check data, (image, label)
        data = train_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 2)
        image, label = data
        # check image, tensor (3, 512, 512)
        self.assertEqual(image.shape, (3, 512, 512))
        self.assertEqual(image.dtype, torch.float32)
        # check label, [0, 3)
        self.assertIsInstance(label, int)
        self.assertGreaterEqual(label, 0)
        self.assertLess(label, 3)
