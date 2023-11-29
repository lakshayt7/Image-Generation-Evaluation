#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np

from scipy.misc import imread
import tensorflow as tf

from src.fid_score import *

import torch 


from clip_score import ImageDirEvaluator

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gen_image_path", default="", type=str,
        help='path to generated images')
    parser.add_argument("--gt_images", default="", type=str,
        help='path to images used for finetuning dreambooth used for calculating clip images')    
    parser.add_argument("--fid_path", default="", type=str,
        help='path to fid statistics taken from some existing dataset probably CelebA')
    parser.add_argument("--control_path", default="", type=str,
        help='path to ground truth control poses')
    parser.add_argument("--text_prompt", default="", type=str,
        help='text prompt used to generate images')

    parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
    parser.add_argument('--num-workers', type=int,
                        help=('Number of processes to use for data loading. '
                            'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                            'By default, uses pool3 features'))
    parser.add_argument('--save-stats', action='store_true',
                        help=('Generate an npz archive from a directory of samples. '
                            'The first path is used as input and the second as output.'))


    IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                        'tif', 'tiff', 'webp'}


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Paths
    image_path = args.gen_image_path # set path to some generated images
    stats_path = args.fid_path #training set statistics
    text_prompt = args.text_prompt 
    gt_path = args.gt_images
    inception_path = fid.check_or_download_inception(None) # download inception network


    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    if args.save_stats:
        save_fid_stats(args.path, args.batch_size, device, args.dims, num_workers)
   

    fid_value = calculate_fid_given_paths([image_path,stats_path],
                                          args.batch_size,
                                          device,
                                          args.dims,
                                          num_workers)
    print("-------------------------------- FID Score --------------------------------")
    print('FID: ', fid_value)

    clip_evaluator = ImageDirEvaluator(device = device)
    img_score, txt_score = clip_evaluator.evaluate(image_path, gt_path, text_prompt)

    print("------------------------ CLIP Similarity Scores ---------------------------")
    print(f'Image similarity score (important) {img_score}')
    print(f'Text similarity score (may not be relevant for dreambooth) {txt_score}')

