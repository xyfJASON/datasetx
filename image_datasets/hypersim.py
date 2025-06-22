import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset


class Hypersim(Dataset):
    """The Hypersim Dataset.

    Please organize the dataset in the following file structure:

    root
    ├── metadata_images_split_scene_v1.csv
    ├── ai_001_001
    ├── ai_001_002
    ├── ...
    └── ai_055_010

    References:
      - https://github.com/apple/ml-hypersim/blob/main/code/python/tools/scene_generate_images_tonemap.py
      - https://github.com/EnVision-Research/Lotus/blob/main/utils/hypersim_dataset.py
      - https://github.com/prs-eth/Marigold/blob/main/script/depth/dataset_preprocess/hypersim/hypersim_util.py

    """

    def __init__(
            self,
            root: str,
            csv_file: str,
            split: str = 'train',
            transforms: Optional[Callable] = None,
    ):
        if split not in ['train', 'valid', 'val', 'test']:
            raise ValueError(f'Invalid split: {split}')
        split = 'val' if split == 'valid' else split
        self.root = os.path.expanduser(root)
        self.transforms = transforms

        self.metadata = []
        data = pd.read_csv(os.path.expanduser(csv_file))
        data = data[(data['included_in_public_release'] == True) & (data['split_partition_name'] == split)]
        for index, row in data.iterrows():
            image_filename = f"frame.{row['frame_id']:04d}.tonemap.jpg"
            illumination_filename = f"frame.{row['frame_id']:04d}.diffuse_illumination.jpg"
            reflectance_filename = f"frame.{row['frame_id']:04d}.diffuse_reflectance.jpg"
            depth_filename = f"frame.{row['frame_id']:04d}.depth_meters.hdf5"
            normal_filename = f"frame.{row['frame_id']:04d}.normal_cam.hdf5"
            image_path = os.path.join(self.root, row['scene_name'], 'images', f'scene_{row["camera_name"]}_final_preview', image_filename)
            illumination_path = os.path.join(self.root, row['scene_name'], 'images', f'scene_{row["camera_name"]}_final_preview', illumination_filename)
            reflectance_path = os.path.join(self.root, row['scene_name'], 'images', f'scene_{row["camera_name"]}_final_preview', reflectance_filename)
            depth_path = os.path.join(self.root, row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_hdf5', depth_filename)
            normal_path = os.path.join(self.root, row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_hdf5', normal_filename)
            self.metadata.append({
                'scene_name': row['scene_name'],
                'camera_name': row['camera_name'],
                'frame_id': row['frame_id'],
                'image_path': image_path,
                'illumination_path': illumination_path,
                'reflectance_path': reflectance_path,
                'depth_path': depth_path,
                'normal_path': normal_path,
            })

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        metadata = self.metadata[index]

        # read image, illumination, reflectance
        image = Image.open(metadata['image_path']).convert('RGB')
        illumination = Image.open(metadata['illumination_path']).convert('RGB')
        reflectance = Image.open(metadata['reflectance_path']).convert('RGB')

        # read depth
        depth_fd = h5py.File(metadata['depth_path'], 'r')
        dist = np.array(depth_fd['dataset']).astype(float)
        if np.isnan(dist).any():
            raise ValueError(f"NaN in depth file: {metadata['depth_path']}")
        depth = self.dist_2_depth(dist)
        # depth to disparity
        disparity = self.depth_2_disparity(depth)

        # read normal
        normal_fd = h5py.File(metadata['normal_path'], 'r')
        normal = np.array(normal_fd['dataset']).astype(float)
        if np.isnan(normal).any():
            raise ValueError(f"NaN in normal file: {metadata['normal_path']}")
        normal[:,:,0] *= -1
        H, W = normal.shape[:2]
        normal = self.align_normals(normal, depth, [886.81, 886.81, W/2, H/2], H, W)

        # transform to torch tensors
        if self.transforms is not None:
            image = self.transforms(image)
            illumination = self.transforms(illumination)
            reflectance = self.transforms(reflectance)
        depth = torch.from_numpy(depth).float()
        disparity = torch.from_numpy(disparity).float()
        normal = torch.from_numpy(normal).permute(2, 0, 1).float().clamp(-1, 1)

        return dict(
            scene_name=metadata['scene_name'],
            camera_name=metadata['camera_name'],
            frame_id=metadata['frame_id'],
            image=image,
            illumination=illumination,
            reflectance=reflectance,
            depth=depth,
            disparity=disparity,
            normal=normal,
        )

    @staticmethod
    def dist_2_depth(distance: np.ndarray, width: int = 1024, height: int = 768, flt_focal: float = 886.81):
        """According to https://github.com/apple/ml-hypersim/issues/9"""
        img_plane_x = (
            np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width)
            .reshape(1, width)
            .repeat(height, 0)
            .astype(np.float32)[:, :, None]
        )
        img_plane_y = (
            np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height)
            .reshape(height, 1)
            .repeat(width, 1)
            .astype(np.float32)[:, :, None]
        )
        img_plane_z = np.full([height, width, 1], flt_focal, np.float32)
        img_plane = np.concatenate([img_plane_x, img_plane_y, img_plane_z], 2)
        depth = distance / np.linalg.norm(img_plane, 2, 2) * flt_focal
        return depth

    @staticmethod
    def depth_2_disparity(depth: np.ndarray, truncnorm_min: float = 0.02):
        disparity = 1 / depth
        disparity_min = np.quantile(disparity, truncnorm_min)
        disparity_max = np.quantile(disparity, 1 - truncnorm_min)
        disparity_norm = (disparity - disparity_min) / (disparity_max - disparity_min + 1e-5)  # type: ignore
        disparity_norm = np.clip(disparity_norm, 0, 1)
        return disparity_norm

    @staticmethod
    def creat_uv_mesh(H: int, W: int):
        y, x = np.meshgrid(np.arange(0, H, dtype=np.float64), np.arange(0, W, dtype=np.float64), indexing='ij')
        meshgrid = np.stack((x,y))
        ones = np.ones((1,H*W), dtype=np.float64)
        xy = meshgrid.reshape(2, -1)
        return np.concatenate([xy, ones], axis=0)

    def align_normals(self, normal: np.ndarray, depth: np.ndarray, K: list, H: int, W: int):
        """
        Orientation of surface normals in hypersim is not always consistent
        see https://github.com/apple/ml-hypersim/issues/26
        """
        # inv K
        K = np.array([[K[0],    0, K[2]],
                      [   0, K[1], K[3]],
                      [   0,    0,    1]])
        inv_K = np.linalg.inv(K)
        # reprojection depth to camera points
        xy = self.creat_uv_mesh(H, W)
        points = np.matmul(inv_K[:3, :3], xy).reshape(3, H, W)
        points = depth * points
        points = points.transpose((1,2,0))
        # align normal
        orient_mask = np.sum(normal * points, axis=2) < 0
        normal[orient_mask] *= -1
        return normal
