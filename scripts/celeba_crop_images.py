"""Crop CelebA images to squared images and resize them to the desired size.

The code follows the StyleGAN-like cropping strategy, see:
    https://github.com/NVlabs/stylegan/blob/master/dataset_tool.py#L484-L499

"""

import os
import argparse
from tqdm import tqdm

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.datasets as dset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='root directory of CelebA dataset')
    parser.add_argument('--img_size', type=int, required=True, help='output image size')
    parser.add_argument('--ext', type=str, choices=['jpg', 'png'], default='png', help='output image extension')
    args = parser.parse_args()

    cx, cy = 89, 121
    transform = T.Compose([
        T.Lambda(lambda x: TF.crop(x, top=cy-64, left=cx-64, height=128, width=128)),
        T.Resize((args.img_size, args.img_size), antialias=True),
    ])
    for split in ['train', 'valid', 'test']:
        dataset = dset.CelebA(root=args.root, split=split, transform=transform)
        os.makedirs(os.path.join(args.root, f'stylegan-like-{args.img_size}-{args.ext}/{split}'), exist_ok=True)
        for i, img in enumerate(tqdm(dataset)):
            img[0].save(os.path.join(args.root, f'stylegan-like-{args.img_size}-{args.ext}/{split}/{i}.{args.ext}'))
