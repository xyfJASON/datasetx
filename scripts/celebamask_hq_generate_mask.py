"""Generate single color mask from annotated masks.

Adapted from:
 - https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_mask.py
 - https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_color.py
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp


label_list = [
    'skin', 'nose', 'eye_g',
    'l_eye', 'r_eye', 'l_brow',
    'r_brow', 'l_ear', 'r_ear',
    'mouth', 'u_lip', 'l_lip',
    'hair', 'hat', 'ear_r',
    'neck_l', 'neck', 'cloth',
]
color_list = [
    [0, 0, 0],
    [204, 0, 0], [76, 153, 0], [204, 204, 0],
    [51, 51, 255], [204, 0, 204], [0, 255, 255],
    [255, 204, 204], [102, 51, 0], [255, 0, 0],
    [102, 204, 0], [255, 255, 0], [0, 0, 153],
    [0, 0, 204], [255, 51, 153], [0, 204, 204],
    [0, 51, 0], [255, 153, 51], [0, 204, 0],
]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Root directory of CelebAMask-HQ')
    parser.add_argument('--n_processes', type=int, help='Number of processes. Default to be cpu numbers')
    return parser


def func(k):
    folder_num = k // 2000
    im_base = np.zeros((512, 512))
    im_color = np.zeros((512, 512, 3))
    # Generate index image
    for idx, label in enumerate(label_list):
        filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
        if os.path.exists(filename):
            im = np.array(Image.open(filename))
            im = im[:, :, 0]
            im_base[im != 0] = (idx + 1)
    # Generate color mask
    for idx, color in enumerate(color_list):
        im_color[im_base == idx] = color
    # Save generated index and mask
    Image.fromarray(im_base.astype(np.uint8)).save(os.path.join(folder_save_mask, str(k) + '.png'))
    Image.fromarray(im_color.astype(np.uint8)).save(os.path.join(folder_save_color, str(k) + '.png'))


if __name__ == '__main__':
    # Arguments
    args = get_parser().parse_args()
    args.root = os.path.expanduser(args.root)
    args.n_processes = args.n_processes or mp.cpu_count()
    print(f'Using {args.n_processes} processes')

    # Create a new directory `root/CelebAMask-HQ-mask-color` to store the color masks
    folder_base = os.path.join(args.root, 'CelebAMask-HQ-mask-anno')
    folder_save_mask = os.path.join(args.root, 'CelebAMask-HQ-mask')
    folder_save_color = os.path.join(args.root, 'CelebAMask-HQ-mask-color')
    if not os.path.isdir(folder_base):
        raise NotADirectoryError(f'{folder_base} is not an existing directory')
    os.makedirs(folder_save_mask, exist_ok=True)
    os.makedirs(folder_save_color, exist_ok=True)

    # Multiprocessing
    mp.set_start_method("fork")
    pool = mp.Pool(processes=args.n_processes)
    for _ in tqdm(pool.imap(func, range(30000)), total=30000):
        pass
    pool.close()
    pool.join()
    print('Done')
