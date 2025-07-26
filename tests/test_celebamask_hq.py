import random
import unittest

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor

from datasetx import CelebAMaskHQ


class TestCelebAHQ(unittest.TestCase):

    root = '~/data/CelebAMask-HQ'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing CelebAMask-HQ dataset...' + '\033[0m')

    def check_sample(self, sample: dict, size_image: int = 1024, size_mask: int = 512):
        # check sample, {image, mask, mask_color}
        self.assertIsInstance(sample, dict)
        self.assertEqual(sample.keys(), {'image', 'mask', 'mask_color'})
        image, mask, mask_color = sample['image'], sample['mask'], sample['mask_color']
        # check image
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, size_image, size_image))
        self.assertEqual(image.dtype, torch.float32)
        # check mask
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, (size_mask, size_mask))
        self.assertEqual(mask.dtype, torch.int64)
        # check mask_color
        self.assertIsInstance(mask_color, Tensor)
        self.assertEqual(mask_color.shape, (3, size_mask, size_mask))
        self.assertEqual(mask_color.dtype, torch.float32)

    def test_train_split(self):
        train_set = CelebAMaskHQ(self.root, split='train')
        self.assertEqual(len(train_set), 24183)
        self.check_sample(train_set[0])

    def test_valid_split(self):
        valid_set = CelebAMaskHQ(self.root, split='valid')
        self.assertEqual(len(valid_set), 2993)
        self.check_sample(valid_set[0])

    def test_test_split(self):
        test_set = CelebAMaskHQ(self.root, split='test')
        self.assertEqual(len(test_set), 2824)
        self.check_sample(test_set[0])

    def test_all_split(self):
        all_set = CelebAMaskHQ(self.root, split='all')
        self.assertEqual(len(all_set), 30000)
        self.check_sample(all_set[0])

    def test_transform_fn(self):
        train_set = CelebAMaskHQ(
            root=self.root,
            split='train',
            transform_fn=CelebAMaskHQTransform(size=256, flip_prob=0.5, mean=0.5, std=0.5),
        )
        self.check_sample(train_set[0], size_image=256, size_mask=256)


class CelebAMaskHQTransform:
    def __init__(self, size, flip_prob, mean, std):
        self.size = [size, size]
        self.flip_prob = flip_prob
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict):
        image, mask, mask_color = sample['image'], sample['mask'], sample['mask_color']
        # resize
        image = TF.resize(image, self.size, antialias=True)
        mask = TF.resize(mask.unsqueeze(0), self.size, interpolation=T.InterpolationMode.NEAREST).squeeze(0)
        mask_color = TF.resize(mask_color, self.size, interpolation=T.InterpolationMode.NEAREST)
        # random horizontal flip
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            mask_color = TF.hflip(mask_color)
        # normalize
        image = TF.normalize(image, mean=self.mean, std=self.std)
        mask_color = TF.normalize(mask_color, mean=self.mean, std=self.std)
        sample = {'image': image, 'mask': mask, 'mask_color': mask_color}
        return sample
