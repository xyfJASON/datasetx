# Hypersim

[Official website](https://github.com/apple/ml-hypersim) | [Papers with Code](https://paperswithcode.com/dataset/hypersim)

## Brief introduction

> Copied from paperswithcode.

For many fundamental scene understanding tasks, it is difficult or impossible to obtain per-pixel ground truth labels from real images. Hypersim is a photorealistic synthetic dataset for holistic indoor scene understanding. It contains 77,400 images of 461 indoor scenes with detailed per-pixel labels and corresponding ground truth geometry.

## Statistics

**Numbers**: 74,619

**Splits** (train / valid / test): 59,543 / 7,386 / 7,690

**Resolution**: 1024×768

## Usage

### Download

Download and unzip the dataset using [this script](https://github.com/apple/ml-hypersim/blob/main/code/python/tools/dataset_download_images.py) and download the csv file from [here](https://github.com/apple/ml-hypersim/blob/main/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv).

### File structure

Please organize the downloaded dataset in the following file structure:

```text
root
├── metadata_images_split_scene_v1.csv
├── ai_001_001
│   ├── _detail
│   │   ├── cam_00
│   │   ├── mesh
│   │   ├── metadata_cameras.csv
│   │   ├── metadata_node_strings.csv
│   │   ├── metadata_nodes.csv
│   │   └── metadata_scene.csv
│   └── images
│       ├── scene_cam_01_final_hdf5
│       │   ├── frame.0000.color.hdf5
│       │   ├── frame.0000.diffuse_illumination.hdf5
│       │   ├── frame.0000.diffuse_reflectance.hdf5
│       │   ├── frame.0000.residual.hdf5
│       │   └── ...
│       ├── scene_cam_01_final_preview
│       │   ├── frame.0000.color.jpg
│       │   ├── frame.0000.diff.jpg
│       │   ├── frame.0000.diffuse_illumination.jpg
│       │   ├── frame.0000.diffuse_reflectance.jpg
│       │   ├── frame.0000.gamma.jpg
│       │   ├── frame.0000.lambertian.jpg
│       │   ├── frame.0000.non_lambertian.jpg
│       │   ├── frame.0000.residual.jpg
│       │   ├── frame.0000.tonemap.jpg
│       │   └── ...
│       ├── scene_cam_01_geometry_hdf5
│       │   ├── frame.0000.depth_meters.hdf5
│       │   ├── frame.0000.normal_bump_cam.hdf5
│       │   ├── frame.0000.normal_bump_world.hdf5
│       │   ├── frame.0000.normal_cam.hdf5
│       │   ├── frame.0000.normal_world.hdf5
│       │   ├── frame.0000.position.hdf5
│       │   ├── frame.0000.render_entity_id.hdf5
│       │   ├── frame.0000.semantic.hdf5
│       │   ├── frame.0000.semantic_instance.hdf5
│       │   ├── frame.0000.tex_coord.hdf5
│       │   └── ...
│       └── scene_cam_01_geometry_preview
│           ├── frame.0000.color.jpg
│           ├── frame.0000.depth_meters.png
│           ├── frame.0000.gamma.jpg
│           ├── frame.0000.normal_bump_cam.png
│           ├── frame.0000.normal_bump_world.png
│           ├── frame.0000.normal_cam.png
│           ├── frame.0000.normal_world.png
│           ├── frame.0000.render_entity_id.png
│           ├── frame.0000.semantic.png
│           ├── frame.0000.semantic_instance.png
│           ├── frame.0000.tex_coord.png
│           └── ...
├── ai_001_002
├── ...
└── ai_055_010
```

### Filter invalid data

Filter NaN values and black images from the dataset. The script will create a new CSV file with the filtered data.

```shell
python scripts/hypersim_filter_invalid.py \
  --dataroot /path/to/downloaded/hypersim/dataroot \
  --csv_file /path/to/downloaded/hypersim/csv_file \
  --output_csv_file /path/to/hypersim/output_csv_file
```

### Example

```python
from image_datasets import Hypersim

root = '~/data/Hypersim'  # path to the dataset
csv_file = '~/data/Hypersim/metadata_images_split_scene_v1_filtered.csv'  # path to the metadata CSV file
train_set = Hypersim(root=root, csv_file=csv_file, split='train')
valid_set = Hypersim(root=root, csv_file=csv_file, split='valid')
test_set = Hypersim(root=root, csv_file=csv_file, split='test')
print(len(train_set))  # 39449
print(len(valid_set))  # 4289
print(len(test_set))   # 5238
print(train_set[0])
```
