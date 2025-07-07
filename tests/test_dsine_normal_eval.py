import unittest

import torch
from torch import Tensor

from image_datasets import DSINENormalEval
from image_datasets.dsine_normal_eval import Compose, Resize, Normalize


class TestDSINENormalEval(unittest.TestCase):

    root = '~/data/dsine_eval'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing DSINENormalEval dataset...' + '\033[0m')

    def test_nyuv2(self):
        dataset = DSINENormalEval(self.root, dataset='nyuv2')
        # check length
        self.assertEqual(len(dataset), 654)
        # check data, (image, normal, mask)
        data = dataset[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, normal, mask = data
        # check image, tensor (3, 480, 640)
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, 480, 640))
        self.assertEqual(image.dtype, torch.float32)
        # check normal, tensor (3, 480, 640)
        self.assertIsInstance(normal, Tensor)
        self.assertEqual(normal.shape, (3, 480, 640))
        self.assertEqual(normal.dtype, torch.float32)
        # check mask, tensor (1, 480, 640)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, (1, 480, 640))
        self.assertEqual(mask.dtype, torch.bool)

    def test_scannet(self):
        dataset = DSINENormalEval(self.root, dataset='scannet')
        # check length
        self.assertEqual(len(dataset), 300)
        # check data, (image, normal, mask)
        data = dataset[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, normal, mask = data
        # check image, tensor (3, 480, 640)
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, 480, 640))
        self.assertEqual(image.dtype, torch.float32)
        # check normal, tensor (3, 480, 640)
        self.assertIsInstance(normal, Tensor)
        self.assertEqual(normal.shape, (3, 480, 640))
        self.assertEqual(normal.dtype, torch.float32)
        # check mask, tensor (1, 480, 640)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, (1, 480, 640))
        self.assertEqual(mask.dtype, torch.bool)

    def test_ibims(self):
        dataset = DSINENormalEval(self.root, dataset='ibims')
        # check length
        self.assertEqual(len(dataset), 100)
        # check data, (image, normal, mask)
        data = dataset[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, normal, mask = data
        # check image, tensor (3, 480, 640)
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, 480, 640))
        self.assertEqual(image.dtype, torch.float32)
        # check normal, tensor (3, 480, 640)
        self.assertIsInstance(normal, Tensor)
        self.assertEqual(normal.shape, (3, 480, 640))
        self.assertEqual(normal.dtype, torch.float32)
        # check mask, tensor (1, 480, 640)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, (1, 480, 640))
        self.assertEqual(mask.dtype, torch.bool)

    def test_sintel(self):
        dataset = DSINENormalEval(self.root, dataset='sintel')
        # check length
        self.assertEqual(len(dataset), 1064)
        # check data, (image, normal, mask)
        data = dataset[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, normal, mask = data
        # check image, tensor (3, 436, 1024)
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, 436, 1024))
        self.assertEqual(image.dtype, torch.float32)
        # check normal, tensor (3, 436, 1024)
        self.assertIsInstance(normal, Tensor)
        self.assertEqual(normal.shape, (3, 436, 1024))
        self.assertEqual(normal.dtype, torch.float32)
        # check mask, tensor (1, 436, 1024)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, (1, 436, 1024))
        self.assertEqual(mask.dtype, torch.bool)

    def test_vkitti(self):
        dataset = DSINENormalEval(self.root, dataset='vkitti')
        # check length
        self.assertEqual(len(dataset), 1000)
        # check data, (image, normal, mask)
        data = dataset[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, normal, mask = data
        # check image, tensor (3, 375, 1242)
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, 375, 1242))
        self.assertEqual(image.dtype, torch.float32)
        # check normal, tensor (3, 375, 1242)
        self.assertIsInstance(normal, Tensor)
        self.assertEqual(normal.shape, (3, 375, 1242))
        self.assertEqual(normal.dtype, torch.float32)
        # check mask, tensor (1, 375, 1242)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, (1, 375, 1242))
        self.assertEqual(mask.dtype, torch.bool)

    def test_oasis(self):
        dataset = DSINENormalEval(self.root, dataset='oasis')
        # check length
        self.assertEqual(len(dataset), 10000)
        # check data, (image, normal, mask)
        data = dataset[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, normal, mask = data
        # check image
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.dtype, torch.float32)
        # check normal
        self.assertIsInstance(normal, Tensor)
        self.assertEqual(normal.dtype, torch.float32)
        # check mask
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.dtype, torch.bool)

    def test_oasis_with_transforms(self):
        transforms = Compose([
            Resize((480, 640)),
            Normalize(mean=0.5, std=0.5),
        ])
        dataset = DSINENormalEval(
            root=self.root,
            dataset='oasis',
            transforms=transforms,
        )
        # check data, (image, normal, mask)
        data = dataset[0]
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 3)
        image, normal, mask = data
        # check image, tensor (3, 480, 640)
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.shape, (3, 480, 640))
        self.assertEqual(image.dtype, torch.float32)
        # check normal, tensor (3, 480, 640)
        self.assertIsInstance(normal, Tensor)
        self.assertEqual(normal.shape, (3, 480, 640))
        self.assertEqual(normal.dtype, torch.float32)
        # check mask, tensor (1, 480, 640)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.shape, (1, 480, 640))
        self.assertEqual(mask.dtype, torch.bool)
