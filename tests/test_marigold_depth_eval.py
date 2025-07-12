import unittest

import torch
from torch import Tensor

from image_datasets import MarigoldDepthEval
from image_datasets.marigold_depth_eval import Compose, Resize, Normalize


class TestMarigoldDepthEval(unittest.TestCase):

    root = '~/data/marigold'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing MarigoldDepthEval dataset...' + '\033[0m')

    def test_nyuv2(self):
        dataset = MarigoldDepthEval(self.root, dataset='nyuv2')
        # check length
        self.assertEqual(len(dataset), 654)
        # check data, (image, depth, mask)
        data = dataset[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, depth, mask = data
        # check image, tensor (3, 480, 640)
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, 480, 640))
        self.assertEqual(image.dtype, torch.float32)
        # check depth, tensor (1, 480, 640)
        self.assertIsInstance(depth, Tensor)
        self.assertEqual(depth.shape, (1, 480, 640))
        self.assertEqual(depth.dtype, torch.float32)
        # check mask, tensor (1, 480, 640)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, (1, 480, 640))
        self.assertEqual(mask.dtype, torch.bool)

    def test_kitti(self):
        dataset = MarigoldDepthEval(self.root, dataset='kitti')
        # check length
        self.assertEqual(len(dataset), 652)
        # check data, (image, depth, mask)
        data = dataset[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, depth, mask = data
        # check image, tensor (3, 352, 1216)
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, 352, 1216))
        self.assertEqual(image.dtype, torch.float32)
        # check depth, tensor (1, 352, 1216)
        self.assertIsInstance(depth, Tensor)
        self.assertEqual(depth.shape, (1, 352, 1216))
        self.assertEqual(depth.dtype, torch.float32)
        # check mask, tensor (1, 352, 1216)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, (1, 352, 1216))
        self.assertEqual(mask.dtype, torch.bool)

    def test_eth3d(self):
        dataset = MarigoldDepthEval(self.root, dataset='eth3d')
        # check length
        self.assertEqual(len(dataset), 454)
        # check data, (image, depth, mask)
        data = dataset[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, depth, mask = data
        # check image, tensor (3, 4032, 6048)
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, 4032, 6048))
        self.assertEqual(image.dtype, torch.float32)
        # check depth, tensor (1, 4032, 6048)
        self.assertIsInstance(depth, Tensor)
        self.assertEqual(depth.shape, (1, 4032, 6048))
        self.assertEqual(depth.dtype, torch.float32)
        # check mask, tensor (1, 4032, 6048)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, (1, 4032, 6048))
        self.assertEqual(mask.dtype, torch.bool)

    def test_scannet(self):
        dataset = MarigoldDepthEval(self.root, dataset='scannet')
        # check length
        self.assertEqual(len(dataset), 800)
        # check data, (image, depth, mask)
        data = dataset[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, depth, mask = data
        # check image, tensor (3, 480, 640)
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, 480, 640))
        self.assertEqual(image.dtype, torch.float32)
        # check depth, tensor (1, 480, 640)
        self.assertIsInstance(depth, Tensor)
        self.assertEqual(depth.shape, (1, 480, 640))
        self.assertEqual(depth.dtype, torch.float32)
        # check mask, tensor (1, 480, 640)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, (1, 480, 640))
        self.assertEqual(mask.dtype, torch.bool)

    def test_diode(self):
        dataset = MarigoldDepthEval(self.root, dataset='diode')
        # check length
        self.assertEqual(len(dataset), 771)
        # check data, (image, depth, mask)
        data = dataset[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, depth, mask = data
        # check image, tensor (3, 768, 1024)
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, 768, 1024))
        self.assertEqual(image.dtype, torch.float32)
        # check depth, tensor (1, 768, 1024)
        self.assertIsInstance(depth, Tensor)
        self.assertEqual(depth.shape, (1, 768, 1024))
        self.assertEqual(depth.dtype, torch.float32)
        # check mask, tensor (1, 768, 1024)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, (1, 768, 1024))
        self.assertEqual(mask.dtype, torch.bool)

    def test_diode_with_transforms(self):
        transforms = Compose([
            Resize((480, 640)),
            Normalize(mean=0.5, std=0.5),
        ])
        dataset = MarigoldDepthEval(
            root=self.root,
            dataset='diode',
            transforms=transforms,
        )
        # check data, (image, depth, mask)
        data = dataset[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, depth, mask = data
        # check image, tensor (3, 480, 640)
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, 480, 640))
        self.assertEqual(image.dtype, torch.float32)
        # check depth, tensor (1, 480, 640)
        self.assertIsInstance(depth, Tensor)
        self.assertEqual(depth.shape, (1, 480, 640))
        self.assertEqual(depth.dtype, torch.float32)
        # check mask, tensor (1, 480, 640)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, (1, 480, 640))
        self.assertEqual(mask.dtype, torch.bool)
