import unittest
from PIL import Image

import torch
import torchvision.transforms as T

from image_datasets import CelebAHQ


class TestCelebAHQ(unittest.TestCase):

    root = '~/data/CelebA-HQ'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing CelebA-HQ dataset...' + '\033[0m')

    def test_train_split(self):
        train_set = CelebAHQ(self.root, split='train')
        # check length
        self.assertEqual(len(train_set), 24183)
        # check image, size 1024x1024
        image = train_set[0]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (1024, 1024))

    def test_valid_split(self):
        valid_set = CelebAHQ(self.root, split='valid')
        # check length
        self.assertEqual(len(valid_set), 2993)
        # check image, size 1024x1024
        image = valid_set[0]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (1024, 1024))

    def test_test_split(self):
        test_set = CelebAHQ(self.root, split='test')
        # check length
        self.assertEqual(len(test_set), 2824)
        # check image, size 1024x1024
        image = test_set[0]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (1024, 1024))

    def test_all_split(self):
        all_set = CelebAHQ(self.root, split='all')
        # check length
        self.assertEqual(len(all_set), 30000)
        # check image, size 1024x1024
        image = all_set[0]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (1024, 1024))

    def test_train_split_with_transforms(self):
        transforms = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        train_set = CelebAHQ(
            root=self.root,
            split='train',
            transforms=transforms,
        )
        # check image, tensor (3, 256, 256)
        image = train_set[0]
        self.assertEqual(image.shape, (3, 256, 256))
        self.assertEqual(image.dtype, torch.float32)
