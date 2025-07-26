import unittest

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor

from datasetx import DSINENormalEval


class TestDSINENormalEval(unittest.TestCase):

    root = '~/data/dsine_eval'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing DSINENormalEval dataset...' + '\033[0m')

    def check_sample(self, sample: dict, shape: tuple[int, int] = None):
        # check sample, {image, normal, mask}
        self.assertIsInstance(sample, dict)
        self.assertEqual(sample.keys(), {'image', 'normal', 'mask'})
        image, normal, mask = sample['image'], sample['normal'], sample['mask']
        # check image, tensor (3, *shape)
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.dtype, torch.float32)
        if shape is not None:
            self.assertEqual(image.shape, (3, *shape))
        # check normal, tensor (3, *shape)
        self.assertIsInstance(normal, Tensor)
        self.assertEqual(normal.dtype, torch.float32)
        if shape is not None:
            self.assertEqual(normal.shape, (3, *shape))
        # check mask, tensor (1, *shape)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.dtype, torch.bool)
        if shape is not None:
            self.assertEqual(mask.shape, (1, *shape))

    def test_nyuv2(self):
        dataset = DSINENormalEval(self.root, dataset='nyuv2')
        self.assertEqual(len(dataset), 654)
        self.check_sample(dataset[0], (480, 640))

    def test_scannet(self):
        dataset = DSINENormalEval(self.root, dataset='scannet')
        self.assertEqual(len(dataset), 300)
        self.check_sample(dataset[0], (480, 640))

    def test_ibims(self):
        dataset = DSINENormalEval(self.root, dataset='ibims')
        self.assertEqual(len(dataset), 100)
        self.check_sample(dataset[0], (480, 640))

    def test_sintel(self):
        dataset = DSINENormalEval(self.root, dataset='sintel')
        self.assertEqual(len(dataset), 1064)
        self.check_sample(dataset[0], (436, 1024))

    def test_vkitti(self):
        dataset = DSINENormalEval(self.root, dataset='vkitti')
        self.assertEqual(len(dataset), 1000)
        self.check_sample(dataset[0], (375, 1242))

    def test_oasis(self):
        dataset = DSINENormalEval(self.root, dataset='oasis')
        self.assertEqual(len(dataset), 10000)
        self.check_sample(dataset[0], None)

    def test_transform_fn(self):
        dataset = DSINENormalEval(
            root=self.root,
            dataset='oasis',
            transform_fn=OASISTransform(size=(480, 640), mean=0.5, std=0.5)
        )
        self.check_sample(dataset[0], (480, 640))


class OASISTransform:
    def __init__(self, size, mean, std):
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict):
        image, normal, mask = sample['image'], sample['normal'], sample['mask']
        # resize
        image = F.interpolate(image.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False, antialias=True).squeeze(0)
        normal = F.interpolate(normal.unsqueeze(0), size=self.size, mode='nearest').squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=self.size, mode='nearest').squeeze(0) > 0.5
        # normalize
        image = TF.normalize(image, mean=self.mean, std=self.std)
        sample = {'image': image, 'normal': normal, 'mask': mask}
        return sample
