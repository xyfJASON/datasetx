import unittest

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor

from datasetx import MarigoldDepthEval


class TestMarigoldDepthEval(unittest.TestCase):

    root = '~/data/marigold'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing MarigoldDepthEval dataset...' + '\033[0m')

    def check_sample(self, sample: dict, shape: tuple[int, int] = None):
        # check sample, {image, depth, mask}
        self.assertIsInstance(sample, dict)
        self.assertEqual(sample.keys(), {'image', 'depth', 'mask'})
        image, depth, mask = sample['image'], sample['depth'], sample['mask']
        # check image, tensor (3, *shape)
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, *shape))
        self.assertEqual(image.dtype, torch.float32)
        # check depth, tensor (1, *shape)
        self.assertIsInstance(depth, Tensor)
        self.assertEqual(depth.shape, (1, *shape))
        self.assertEqual(depth.dtype, torch.float32)
        # check mask, tensor (1, *shape)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, (1, *shape))
        self.assertEqual(mask.dtype, torch.bool)

    def test_nyuv2(self):
        dataset = MarigoldDepthEval(self.root, dataset='nyuv2')
        self.assertEqual(len(dataset), 654)
        self.check_sample(dataset[0], (480, 640))

    def test_kitti(self):
        dataset = MarigoldDepthEval(self.root, dataset='kitti')
        self.assertEqual(len(dataset), 652)
        self.check_sample(dataset[0], (352, 1216))

    def test_eth3d(self):
        dataset = MarigoldDepthEval(self.root, dataset='eth3d')
        self.assertEqual(len(dataset), 454)
        self.check_sample(dataset[0], (4032, 6048))

    def test_scannet(self):
        dataset = MarigoldDepthEval(self.root, dataset='scannet')
        self.assertEqual(len(dataset), 800)
        self.check_sample(dataset[0], (480, 640))

    def test_diode(self):
        dataset = MarigoldDepthEval(self.root, dataset='diode')
        self.assertEqual(len(dataset), 771)
        self.check_sample(dataset[0], (768, 1024))

    def test_transform_fn(self):
        dataset = MarigoldDepthEval(
            root=self.root,
            dataset='diode',
            transform_fn=DIODETransform(size=(480, 640), mean=0.5, std=0.5),
        )
        self.check_sample(dataset[0], (480, 640))


class DIODETransform:
    def __init__(self, size, mean, std):
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict):
        image, depth, mask = sample['image'], sample['depth'], sample['mask']
        # resize
        image = F.interpolate(image.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False, antialias=True).squeeze(0)
        depth = F.interpolate(depth.unsqueeze(0), size=self.size, mode='nearest').squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=self.size, mode='nearest').squeeze(0) > 0.5
        # normalize
        image = TF.normalize(image, mean=self.mean, std=self.std)
        sample = {'image': image, 'depth': depth, 'mask': mask}
        return sample
