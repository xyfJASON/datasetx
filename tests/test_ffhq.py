import unittest
from PIL import Image

import torch
import torchvision.transforms as T

from image_datasets import FFHQ


class TestFFHQ(unittest.TestCase):

    root = '~/data/FFHQ'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing FFHQ dataset...' + '\033[0m')

    def test_train_split(self):
        train_set = FFHQ(self.root, split='train')
        # check length
        self.assertEqual(len(train_set), 60000)
        # check image, size 1024x1024
        image = train_set[0]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (1024, 1024))

    def test_test_split(self):
        test_set = FFHQ(self.root, split='test')
        # check length
        self.assertEqual(len(test_set), 10000)
        # check image, size 1024x1024
        image = test_set[0]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (1024, 1024))

    def test_all_split(self):
        all_set = FFHQ(self.root, split='all')
        # check length
        self.assertEqual(len(all_set), 70000)
        # check image, size 1024x1024
        image = all_set[0]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (1024, 1024))

    def test_thumbnails_version(self):
        all_test = FFHQ(self.root, split='all', version='thumbnails128x128')
        # check length
        self.assertEqual(len(all_test), 70000)
        # check image, size 128x128
        image = all_test[0]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (128, 128))

    def test_train_split_with_transforms(self):
        transforms = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        train_set = FFHQ(
            root=self.root,
            split='train',
            transforms=transforms,
        )
        # check image, tensor (3, 512, 512)
        image = train_set[0]
        self.assertEqual(image.shape, (3, 512, 512))
        self.assertEqual(image.dtype, torch.float32)
