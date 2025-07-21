import unittest
import torch
from image_datasets import Hypersim


class TestHypersim(unittest.TestCase):

    root = '~/data/Hypersim/data_original_extracted'
    csv_file = '~/data/Hypersim/metadata_images_split_scene_v1.csv'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing Hypersim dataset...' + '\033[0m')

    def check_sample(self, sample: dict):
        # check sample, {image, color, illumination, reflectance, residual, depth, disparity, normal}
        self.assertIsInstance(sample, dict)
        # check image, size 768x1024, range [0, 1]
        self.assertIsInstance(sample['image'], torch.Tensor)
        self.assertEqual(sample['image'].shape, (3, 768, 1024))
        self.assertGreaterEqual(sample['image'].min(), 0)
        self.assertLessEqual(sample['image'].max(), 1)
        # check color, size 768x1024, range [0, +inf)
        self.assertIsInstance(sample['color'], torch.Tensor)
        self.assertEqual(sample['color'].shape, (3, 768, 1024))
        self.assertGreaterEqual(sample['color'].min(), 0)
        # check illumination, size 768x1024, range [0, +inf)
        self.assertIsInstance(sample['illumination'], torch.Tensor)
        self.assertEqual(sample['illumination'].shape, (3, 768, 1024))
        self.assertGreaterEqual(sample['illumination'].min(), 0)
        # check reflectance, size 768x1024, range [0, 1]
        self.assertIsInstance(sample['reflectance'], torch.Tensor)
        self.assertEqual(sample['reflectance'].shape, (3, 768, 1024))
        self.assertGreaterEqual(sample['reflectance'].min(), 0)
        self.assertLessEqual(sample['reflectance'].max(), 1)
        # check residual, size 768x1024, range [0, +inf)
        self.assertIsInstance(sample['residual'], torch.Tensor)
        self.assertEqual(sample['residual'].shape, (3, 768, 1024))
        self.assertGreaterEqual(sample['residual'].min(), 0)
        # check depth, size 768x1024, range [0, +inf)
        self.assertIsInstance(sample['depth'], torch.Tensor)
        self.assertEqual(sample['depth'].shape, (768, 1024))
        self.assertGreaterEqual(sample['depth'].min(), 0)
        # check disparity, size 768x1024, range [0, 1]
        self.assertIsInstance(sample['disparity'], torch.Tensor)
        self.assertEqual(sample['disparity'].shape, (768, 1024))
        self.assertGreaterEqual(sample['disparity'].min(), 0)
        self.assertLessEqual(sample['disparity'].max(), 1)
        # check normal, size 768x1024, range [-1, 1]
        self.assertIsInstance(sample['normal'], torch.Tensor)
        self.assertEqual(sample['normal'].shape, (3, 768, 1024))
        self.assertGreaterEqual(sample['normal'].min(), -1)
        self.assertLessEqual(sample['normal'].max(), 1)

    def test_train_split(self):
        train_set = Hypersim(self.root, self.csv_file, split='train')
        self.assertEqual(len(train_set), 59543)
        self.check_sample(train_set[0])

    def test_val_split(self):
        val_set = Hypersim(self.root, self.csv_file, split='val')
        self.assertEqual(len(val_set), 7386)
        self.check_sample(val_set[0])

    def test_test_split(self):
        test_set = Hypersim(self.root, self.csv_file, split='test')
        self.assertEqual(len(test_set), 7690)
        self.check_sample(test_set[0])

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
