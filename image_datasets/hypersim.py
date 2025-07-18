import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


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

    FILES = {
        'image': ('final_preview', 'tonemap.jpg'),
        'color': ('final_hdf5', 'color.hdf5'),
        'illumination': ('final_hdf5', 'diffuse_illumination.hdf5'),
        'reflectance': ('final_hdf5', 'diffuse_reflectance.hdf5'),
        'residual': ('final_hdf5', 'residual.hdf5'),
        'depth': ('geometry_hdf5', 'depth_meters.hdf5'),
        'normal': ('geometry_hdf5', 'normal_cam.hdf5'),
    }

    def __init__(
            self,
            root: str,
            csv_file: str,
            split: str = 'train',
    ):
        if split not in ['train', 'valid', 'val', 'test']:
            raise ValueError(f'Invalid split: {split}')
        split = 'val' if split == 'valid' else split
        self.root = os.path.expanduser(root)

        self.metadata = []
        data = pd.read_csv(os.path.expanduser(csv_file))
        data = data[(data['included_in_public_release'] == True) & (data['split_partition_name'] == split)]
        for _, row in data.iterrows():
            scene_name = row['scene_name']
            camera_name = row['camera_name']
            frame_id = row['frame_id']
            filepaths = {
                f'{k}_path': os.path.join(
                    self.root, scene_name, 'images',
                    f'scene_{camera_name}_{v[0]}',
                    f'frame.{frame_id:04d}.{v[1]}',
                ) for k, v in self.FILES.items()
            }
            self.metadata.append({
                'scene_name': scene_name,
                'camera_name': camera_name,
                'frame_id': frame_id,
                **filepaths,
            })

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        metadata = self.metadata[index]

        # read image
        image = Image.open(metadata['image_path']).convert('RGB')

        # read color, illumination, reflectance, residual
        with h5py.File(metadata['color_path'], 'r') as f:
            color = np.array(f['dataset']).astype(float)
        with h5py.File(metadata['illumination_path'], 'r') as f:
            illumination = np.array(f['dataset']).astype(float)
        with h5py.File(metadata['reflectance_path'], 'r') as f:
            reflectance = np.array(f['dataset']).astype(float)
        with h5py.File(metadata['residual_path'], 'r') as f:
            residual = np.array(f['dataset']).astype(float)

        # read dist, convert to depth and disparity
        with h5py.File(metadata['depth_path'], 'r') as f:
            dist = np.array(f['dataset']).astype(float)
        if np.isnan(dist).any():
            raise ValueError(f"NaN in depth file: {metadata['depth_path']}")
        depth = self.dist_2_depth(dist)
        disparity = self.depth_2_disparity(depth)

        # read normal
        with h5py.File(metadata['normal_path'], 'r') as f:
            normal = np.array(f['dataset']).astype(float)
        if np.isnan(normal).any():
            raise ValueError(f"NaN in normal file: {metadata['normal_path']}")
        normal[:,:,0] *= -1
        H, W = normal.shape[:2]
        normal = self.align_normals(normal, depth, [886.81, 886.81, W/2, H/2], H, W)

        # transform to torch tensors
        image = to_tensor(image)
        color = torch.from_numpy(color).permute(2, 0, 1).float()
        illumination = torch.from_numpy(illumination).permute(2, 0, 1).float()
        reflectance = torch.from_numpy(reflectance).permute(2, 0, 1).float()
        residual = torch.from_numpy(residual).permute(2, 0, 1).float()
        depth = torch.from_numpy(depth).float()
        disparity = torch.from_numpy(disparity).float()
        normal = torch.from_numpy(normal).permute(2, 0, 1).float().clamp(-1, 1)

        return dict(
            scene_name=metadata['scene_name'],
            camera_name=metadata['camera_name'],
            frame_id=metadata['frame_id'],
            image=image,
            color=color,
            illumination=illumination,
            reflectance=reflectance,
            residual=residual,
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
