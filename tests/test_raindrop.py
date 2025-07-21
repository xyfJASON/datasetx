import random
import unittest

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor

from image_datasets import Raindrop


class TestRaindrop(unittest.TestCase):

    root = '~/data/Raindrop'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing Raindrop dataset...' + '\033[0m')

    def check_sample(self, sample: dict, size: int = None):
        # check sample, {image, gt}
        self.assertIsInstance(sample, dict)
        self.assertEqual(sample.keys(), {'image', 'gt'})
        image, gt = sample['image'], sample['gt']
        # check image
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.dtype, torch.float32)
        if size is not None:
            self.assertEqual(image.shape, (3, size, size))
        # check gt
        self.assertIsInstance(gt, Tensor)
        self.assertEqual(gt.dtype, torch.float32)
        if size is not None:
            self.assertEqual(gt.shape, (3, size, size))

    def test_train_split(self):
        train_set = Raindrop(self.root, split='train')
        self.assertEqual(len(train_set), 861)
        self.check_sample(train_set[0])

    def test_test_a_split(self):
        test_a_set = Raindrop(self.root, split='test_a')
        self.assertEqual(len(test_a_set), 58)
        self.check_sample(test_a_set[0])

    def test_test_b_split(self):
        test_b_set = Raindrop(self.root, split='test_b')
        self.assertEqual(len(test_b_set), 249)
        self.check_sample(test_b_set[0])

    def test_transform_fn(self):
        train_set = Raindrop(
            root=self.root,
            split='train',
            transform_fn=RaindropTransform(size=256, flip_prob=0.5, mean=0.5, std=0.5),
        )
        self.check_sample(train_set[0], size=256)
        # check crop and flip
        for idx in [0, 10, 50, 100, -1]:
            sample = train_set[idx]
            image, gt = sample['image'], sample['gt']
            self.assertLess(((image - gt) ** 2).mean(), 0.1)


class RaindropTransform:
    def __init__(self, size, flip_prob, mean, std):
        self.size = size
        self.flip_prob = flip_prob
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict):
        image, gt = sample['image'], sample['gt']
        # resize
        image = TF.resize(image, self.size, antialias=True)
        gt = TF.resize(gt, self.size, antialias=True)
        # random crop
        image = self.pad_if_smaller(image, self.size)
        gt = self.pad_if_smaller(gt, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = TF.crop(image, *crop_params)
        gt = TF.crop(gt, *crop_params)
        # random horizontal flip
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            gt = TF.hflip(gt)
        # normalize
        image = TF.normalize(image, mean=self.mean, std=self.std)
        gt = TF.normalize(gt, mean=self.mean, std=self.std)
        sample = {'image': image, 'gt': gt}
        return sample

    @staticmethod
    def pad_if_smaller(img, size, fill=0):
        min_size = min(img.shape[1:])
        if min_size < size:
            oh, ow = img.shape[1:]
            padh = size - oh if oh < size else 0
            padw = size - ow if ow < size else 0
            img = TF.pad(img, [0, 0, padw, padh], fill=fill)
        return img
