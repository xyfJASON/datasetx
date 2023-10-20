"""Remove black edges in the images.

Before processing:
    root
    └── portraits           (contains 302,652 images)
        ├── 10000310.jpg
        ├── ...
        └── 9999900.jpg

After processing:
    root
    ├── portraits           (contains 232,885 images under default arguments)
    │   ├── 10000310.jpg
    │   ├── ...
    │   └── 9999800.jpg
    └── portraits-backup    (contains 302,652 images)
        ├── 10000310.jpg
        ├── ...
        └── 9999900.jpg

"""

import os
import glob
import shutil
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Root directory of Danbooru2019Portraits')
    parser.add_argument('--n_processes', type=int, help='Number of processes. Default to be cpu numbers')
    parser.add_argument('--portion', type=float, default=0.075,
                        help='Only crop and save images whose black edges width / image width < portion')
    parser.add_argument('--threshold', type=int, default=50,
                        help='Pixel values less than threshold are viewed as black')
    return parser


def count_edge(img):
    up, down, left, right = 0, 0, 0, 0
    while up < img.shape[0] and np.mean(img[up, :, :]) <= args.threshold:
        up += 1
    while down < img.shape[0] and np.mean(img[img.shape[0] - 1 - down, :, :]) <= args.threshold:
        down += 1
    while left < img.shape[1] and np.mean(img[:, left, :]) <= args.threshold:
        left += 1
    while right < img.shape[1] and np.mean(img[:, img.shape[1] - 1 - right, :]) <= args.threshold:
        right += 1
    return up, down, left, right


def func(img_path):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    u, d, l, r = count_edge(img)
    o = max(u, d, l, r)
    if o / img.shape[0] < args.portion:
        img = img[o:img.shape[0] - o, o:img.shape[1] - o, ...]
        img = Image.fromarray(img)
        img.save(img_path.replace('portraits', 'portraits-tmp'))


if __name__ == '__main__':
    # Arguments
    args = get_parser().parse_args()
    args.root = os.path.expanduser(args.root)
    args.n_processes = args.n_processes or mp.cpu_count()
    print(f'Using {args.n_processes} processes')

    # Create a new directory `root/portraits-tmp` to store the renamed images
    if not os.path.isdir(os.path.join(args.root, 'portraits')):
        raise NotADirectoryError(f'`{os.path.join(args.root, "portraits")}` does not exist')
    os.makedirs(os.path.join(args.root, 'portraits-tmp'), exist_ok=True)
    img_paths = glob.glob(os.path.join(args.root, 'portraits', '*.jpg'))

    # Multiprocessing
    mp.set_start_method("fork")
    pool = mp.Pool(processes=args.n_processes)
    for _ in tqdm(pool.imap(func, img_paths), total=len(img_paths)):
        pass
    pool.close()
    pool.join()

    # Backup the original directory and rename the temporary directory
    shutil.move(os.path.join(args.root, 'portraits'), os.path.join(args.root, 'portraits-backup'))
    shutil.move(os.path.join(args.root, 'portraits-tmp'), os.path.join(args.root, 'portraits'))
    print('Done')
