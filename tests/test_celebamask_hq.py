import unittest
from PIL import Image

import torch

from image_datasets import CelebAMaskHQ
from image_datasets.celebamask_hq import Compose, Resize, ToTensor, Normalize


class TestCelebAHQ(unittest.TestCase):

    root = '~/data/CelebAMask-HQ'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing CelebAMask-HQ dataset...' + '\033[0m')

    def test_train_split(self):
        train_set = CelebAMaskHQ(self.root, split='train')
        # check length
        self.assertEqual(len(train_set), 24183)
        # check data, (image, mask, mask_color)
        data = train_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, mask, mask_color = data
        # check image, size 1024x1024
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (1024, 1024))
        # check mask, size 512x512
        self.assertIsInstance(mask, Image.Image)
        self.assertEqual(mask.size, (512, 512))
        # check mask_color, size 512x512
        self.assertIsInstance(mask_color, Image.Image)
        self.assertEqual(mask_color.size, (512, 512))

    def test_valid_split(self):
        valid_set = CelebAMaskHQ(self.root, split='valid')
        # check length
        self.assertEqual(len(valid_set), 2993)
        # check data, (image, mask, mask_color)
        data = valid_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, mask, mask_color = data
        # check image, size 1024x1024
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (1024, 1024))
        # check mask, size 512x512
        self.assertIsInstance(mask, Image.Image)
        self.assertEqual(mask.size, (512, 512))
        # check mask_color, size 512x512
        self.assertIsInstance(mask_color, Image.Image)
        self.assertEqual(mask_color.size, (512, 512))

    def test_test_split(self):
        test_set = CelebAMaskHQ(self.root, split='test')
        # check length
        self.assertEqual(len(test_set), 2824)
        # check data, (image, mask, mask_color)
        data = test_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, mask, mask_color = data
        # check image, size 1024x1024
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (1024, 1024))
        # check mask, size 512x512
        self.assertIsInstance(mask, Image.Image)
        self.assertEqual(mask.size, (512, 512))
        # check mask_color, size 512x512
        self.assertIsInstance(mask_color, Image.Image)
        self.assertEqual(mask_color.size, (512, 512))

    def test_all_split(self):
        all_set = CelebAMaskHQ(self.root, split='all')
        # check length
        self.assertEqual(len(all_set), 30000)
        # check data, (image, mask, mask_color)
        data = all_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, mask, mask_color = data
        # check image, size 1024x1024
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (1024, 1024))
        # check mask, size 512x512
        self.assertIsInstance(mask, Image.Image)
        self.assertEqual(mask.size, (512, 512))
        # check mask_color, size 512x512
        self.assertIsInstance(mask_color, Image.Image)
        self.assertEqual(mask_color.size, (512, 512))

    def test_train_split_with_transforms(self):
        transforms = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize(mean=0.5, std=0.5),
        ])
        train_set = CelebAMaskHQ(
            root=self.root,
            split='train',
            transforms=transforms,
        )
        # check data, (image, mask, mask_color)
        data = train_set[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, mask, mask_color = data
        # check image, tensor (3, 256, 256)
        self.assertEqual(image.shape, (3, 256, 256))
        self.assertEqual(image.dtype, torch.float32)
        # check mask, tensor (256, 256)
        self.assertEqual(mask.shape, (256, 256))
        self.assertEqual(mask.dtype, torch.int64)
        # check mask_color, tensor (3, 256, 256)
        self.assertEqual(mask_color.shape, (3, 256, 256))
        self.assertEqual(mask_color.dtype, torch.float32)
