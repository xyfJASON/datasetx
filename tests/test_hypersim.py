import unittest

import torch
from image_datasets import Hypersim


class TestHypersim(unittest.TestCase):

    root = '~/data/Hypersim/data_original_extracted'
    csv_file = '~/data/Hypersim/metadata_images_split_scene_v1.csv'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing Hypersim dataset...' + '\033[0m')

    def test_train_split(self):
        train_set = Hypersim(self.root, self.csv_file, split='train')
        # check length
        self.assertEqual(len(train_set), 59543)
        # check data, dict(image, color, illumination, reflectance, residual, depth, disparity, normal)
        data = train_set[0]
        self.assertIsInstance(data, dict)
        # check image, size 768x1024
        self.assertIsInstance(data['image'], torch.Tensor)
        self.assertEqual(data['image'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['image'].min(), 0)
        self.assertLessEqual(data['image'].max(), 1)
        # check color, size 768x1024
        self.assertIsInstance(data['color'], torch.Tensor)
        self.assertEqual(data['color'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['color'].min(), 0)
        # check illumination, size 768x1024
        self.assertIsInstance(data['illumination'], torch.Tensor)
        self.assertEqual(data['illumination'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['illumination'].min(), 0)
        # check reflectance, size 768x1024
        self.assertIsInstance(data['reflectance'], torch.Tensor)
        self.assertEqual(data['reflectance'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['reflectance'].min(), 0)
        self.assertLessEqual(data['reflectance'].max(), 1)
        # check residual, size 768x1024
        self.assertIsInstance(data['residual'], torch.Tensor)
        self.assertEqual(data['residual'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['residual'].min(), 0)
        # check depth, size 768x1024
        self.assertIsInstance(data['depth'], torch.Tensor)
        self.assertEqual(data['depth'].shape, (768, 1024))
        self.assertGreaterEqual(data['depth'].min(), 0)
        # check disparity, size 768x1024
        self.assertIsInstance(data['disparity'], torch.Tensor)
        self.assertEqual(data['disparity'].shape, (768, 1024))
        self.assertGreaterEqual(data['disparity'].min(), 0)
        self.assertLessEqual(data['disparity'].max(), 1)
        # check normal, size 768x1024
        self.assertIsInstance(data['normal'], torch.Tensor)
        self.assertEqual(data['normal'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['normal'].min(), -1)
        self.assertLessEqual(data['normal'].max(), 1)

    def test_val_split(self):
        val_set = Hypersim(self.root, self.csv_file, split='val')
        # check length
        self.assertEqual(len(val_set), 7386)
        # check data, dict(image, color, illumination, reflectance, residual, depth, disparity, normal)
        data = val_set[0]
        self.assertIsInstance(data, dict)
        # check image, size 768x1024
        self.assertIsInstance(data['image'], torch.Tensor)
        self.assertEqual(data['image'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['image'].min(), 0)
        self.assertLessEqual(data['image'].max(), 1)
        # check color, size 768x1024
        self.assertIsInstance(data['color'], torch.Tensor)
        self.assertEqual(data['color'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['color'].min(), 0)
        # check illumination, size 768x1024
        self.assertIsInstance(data['illumination'], torch.Tensor)
        self.assertEqual(data['illumination'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['illumination'].min(), 0)
        # check reflectance, size 768x1024
        self.assertIsInstance(data['reflectance'], torch.Tensor)
        self.assertEqual(data['reflectance'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['reflectance'].min(), 0)
        self.assertLessEqual(data['reflectance'].max(), 1)
        # check residual, size 768x1024
        self.assertIsInstance(data['residual'], torch.Tensor)
        self.assertEqual(data['residual'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['residual'].min(), 0)
        # check depth, size 768x1024
        self.assertIsInstance(data['depth'], torch.Tensor)
        self.assertEqual(data['depth'].shape, (768, 1024))
        self.assertGreaterEqual(data['depth'].min(), 0)
        # check disparity, size 768x1024
        self.assertIsInstance(data['disparity'], torch.Tensor)
        self.assertEqual(data['disparity'].shape, (768, 1024))
        self.assertGreaterEqual(data['disparity'].min(), 0)
        self.assertLessEqual(data['disparity'].max(), 1)
        # check normal, size 768x1024
        self.assertIsInstance(data['normal'], torch.Tensor)
        self.assertEqual(data['normal'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['normal'].min(), -1)
        self.assertLessEqual(data['normal'].max(), 1)

    def test_test_split(self):
        test_set = Hypersim(self.root, self.csv_file, split='test')
        # check length
        self.assertEqual(len(test_set), 7690)
        # check data, dict(image, color, illumination, reflectance, residual, depth, disparity, normal)
        data = test_set[0]
        self.assertIsInstance(data, dict)
        # check image, size 768x1024
        self.assertIsInstance(data['image'], torch.Tensor)
        self.assertEqual(data['image'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['image'].min(), 0)
        self.assertLessEqual(data['image'].max(), 1)
        # check color, size 768x1024
        self.assertIsInstance(data['color'], torch.Tensor)
        self.assertEqual(data['color'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['color'].min(), 0)
        # check illumination, size 768x1024
        self.assertIsInstance(data['illumination'], torch.Tensor)
        self.assertEqual(data['illumination'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['illumination'].min(), 0)
        # check reflectance, size 768x1024
        self.assertIsInstance(data['reflectance'], torch.Tensor)
        self.assertEqual(data['reflectance'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['reflectance'].min(), 0)
        self.assertLessEqual(data['reflectance'].max(), 1)
        # check residual, size 768x1024
        self.assertIsInstance(data['residual'], torch.Tensor)
        self.assertEqual(data['residual'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['residual'].min(), 0)
        # check depth, size 768x1024
        self.assertIsInstance(data['depth'], torch.Tensor)
        self.assertEqual(data['depth'].shape, (768, 1024))
        self.assertGreaterEqual(data['depth'].min(), 0)
        # check disparity, size 768x1024
        self.assertIsInstance(data['disparity'], torch.Tensor)
        self.assertEqual(data['disparity'].shape, (768, 1024))
        self.assertGreaterEqual(data['disparity'].min(), 0)
        self.assertLessEqual(data['disparity'].max(), 1)
        # check normal, size 768x1024
        self.assertIsInstance(data['normal'], torch.Tensor)
        self.assertEqual(data['normal'].shape, (3, 768, 1024))
        self.assertGreaterEqual(data['normal'].min(), -1)
        self.assertLessEqual(data['normal'].max(), 1)

    def test_diff(self):
        dataset = Hypersim(self.root, self.csv_file, split='train')
        # check color ~= reflectance * illumination + residual
        for idx in [0, 10, 100, 1000, -1]:
            color = dataset[idx]['color']
            reflectance = dataset[idx]['reflectance']
            illumination = dataset[idx]['illumination']
            residual = dataset[idx]['residual']
            diff = color - (reflectance * illumination + residual)
            self.assertLess(diff.abs().mean(), 5e-3)
