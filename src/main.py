#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np


import torch 


from clip_score import ImageDirEvaluator

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gen_image_path", default="", type=str,
        help='path to generated images')
    parser.add_argument("--gt_images", default="", type=str,
        help='path to images used for finetuning dreambooth used for calculating clip images')    
    parser.add_argument("--control_path", default="", type=str,
        help='path to ground truth control poses')
    parser.add_argument("--text_prompt", default="", type=str,
        help='text prompt used to generate images')

  
    args = parser.parse_args()

    # Paths
    image_path = args.gen_image_path # set path to some generated images
    text_prompt = args.text_prompt 
    gt_path = args.gt_images

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    clip_evaluator = ImageDirEvaluator(device = device)
    img_score, txt_score = clip_evaluator.evaluate(image_path, gt_path, text_prompt)

    print("------------------------ CLIP Similarity Scores ---------------------------")
    print(f'Image similarity score (important) {img_score}')
    print(f'Text similarity score (may not be relevant for dreambooth) {txt_score}')


