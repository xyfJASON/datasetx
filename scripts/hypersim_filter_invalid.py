import os
import h5py
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from PIL import Image
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True, help='Path to the hypersim dataset root directory')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the hypersim CSV file')
    parser.add_argument('--output_csv_file', type=str, required=True, help='Path to the output CSV file')
    parser.add_argument('--n_processes', type=int, help='Number of processes. Default to be cpu numbers')
    return parser


def has_nan(hdf5_path):
    """Check if the HDF5 file contains any NaN values."""
    with h5py.File(hdf5_path, 'r') as file:
        for key in file.keys():
            if np.isnan(file[key][:]).any():
                return True
    return False


def is_black(image_path):
    """Check if the image file is all black, see https://github.com/apple/ml-hypersim/issues/22."""
    image = np.array(Image.open(image_path).convert('RGB'))
    return not image.sum() > 0


def func(iterrow):
    index, row = iterrow
    image_file_name = f"frame.{row['frame_id']:04d}.tonemap.jpg"
    depth_file_name = f"frame.{row['frame_id']:04d}.depth_meters.hdf5"
    normal_file_name = f"frame.{row['frame_id']:04d}.normal_cam.hdf5"
    src_image_path = os.path.join(args.dataroot, row['scene_name'], 'images', f'scene_{row["camera_name"]}_final_preview', image_file_name)
    src_depth_path = os.path.join(args.dataroot, row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_hdf5', depth_file_name)
    src_normal_path = os.path.join(args.dataroot, row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_hdf5', normal_file_name)
    if not (has_nan(src_depth_path) or has_nan(src_normal_path) or is_black(src_image_path)):
        return index, row
    return None


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.n_processes = args.n_processes or mp.cpu_count()
    print(f'Using {args.n_processes} processes')

    data = pd.read_csv(os.path.expanduser(args.csv_file))
    data = data[data['included_in_public_release'] == True]
    new_data = pd.DataFrame(columns=data.columns)

    mp.set_start_method("fork")
    pool = mp.Pool(processes=args.n_processes)
    for res in tqdm(pool.imap(func, data.iterrows()), total=len(data)):
        if res is not None:
            new_data = pd.concat([new_data, res[1].to_frame().T])
    pool.close()
    pool.join()
    new_data.to_csv(args.output_csv_file, index=False)
    print('Done')
