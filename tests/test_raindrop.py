import unittest
from PIL import Image

import torch

from image_datasets import Raindrop
from image_datasets.raindrop import Compose, Resize, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize


class TestRaindrop(unittest.TestCase):

    root = '~/data/Raindrop'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing Raindrop dataset...' + '\033[0m')

    def test_train_split(self):
        train_set = Raindrop(self.root, split='train')
        # check length
        self.assertEqual(len(train_set), 861)
        # check data, (image, ground-truth)
        data = train_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 2)
        image, gt = data
        # check image
        self.assertIsInstance(image, Image.Image)
        # check gt
        self.assertIsInstance(gt, Image.Image)

    def test_test_a_split(self):
        test_a_set = Raindrop(self.root, split='test_a')
        # check length
        self.assertEqual(len(test_a_set), 58)
        # check data, (image, label)
        data = test_a_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 2)
        image, gt = data
        # check image
        self.assertIsInstance(image, Image.Image)
        # check gt
        self.assertIsInstance(gt, Image.Image)

    def test_test_b_split(self):
        test_b_set = Raindrop(self.root, split='test_b')
        # check length
        self.assertEqual(len(test_b_set), 249)
        # check data, (image, label)
        data = test_b_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 2)
        image, gt = data
        # check image
        self.assertIsInstance(image, Image.Image)
        # check gt
        self.assertIsInstance(gt, Image.Image)

    def test_train_split_with_transforms(self):
        transforms = Compose([
            Resize(256),
            RandomCrop(256),
            RandomHorizontalFlip(0.5),
            ToTensor(),
            Normalize(0.5, 0.5),
        ])
        train_set = Raindrop(
            root=self.root,
            split='train',
            transforms=transforms,
        )
        # check data, (image, label)
        data = train_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 2)
        image, gt = data
        # check image, tensor (3, 256, 256)
        self.assertEqual(image.shape, (3, 256, 256))
        self.assertEqual(image.dtype, torch.float32)
        # check gt, tensor (3, 256, 256)
        self.assertEqual(gt.shape, (3, 256, 256))
        self.assertEqual(gt.dtype, torch.float32)
        # check transforms
        self.assertTrue(((image - gt) ** 2).mean() < 0.04)
