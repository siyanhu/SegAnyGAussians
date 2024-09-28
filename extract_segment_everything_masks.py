import os
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
import root_file_io as fio

if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM segment everything masks extracting params")
    
    parser.add_argument("--image_root", default='./data/360_v2/garden/', type=str)
    parser.add_argument("--sam_checkpoint_path", default="./dependencies/sam_ckpt/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--downsample", default="1", type=str)

    args = parser.parse_args()
    
    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)

    # Default settings of SAM
    mask_generator = SamAutomaticMaskGenerator(
        sam, 
        pred_iou_thresh = 0.88, 
        stability_score_thresh = 0.95, 
        min_mask_region_area = 0
    )
    # Trex is hard to segment with the default setting
    # mask_generator = SamAutomaticMaskGenerator(
    #     model=sam,
    #     points_per_side=32,
    #     pred_iou_thresh=0.8,
    #     stability_score_thresh=0.8,
    #     crop_n_layers=0,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=100,  # Requires open-cv to run post-processing
    # )
    if args.downsample == "1":
        IMAGE_DIR = os.path.join(args.image_root, 'images')
    else:
        IMAGE_DIR = os.path.join(args.image_root, 'images_'+args.downsample)
    assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
    OUTPUT_DIR = os.path.join(args.image_root, 'sam_masks')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Extracting SAM segment everything masks...")
    image_seqs = fio.traverse_dir(IMAGE_DIR, full_path=True, towards_sub=False)
    image_seqs = fio.filter_folder(image_seqs, filter_out=False, filter_text='seq')
    image_paths = []
    for seq_dir in image_seqs:
        seq_image_paths = fio.traverse_dir(seq_dir, full_path=True, towards_sub=False)
        seq_image_paths = fio.filter_ext(seq_image_paths, filter_out_target=False, ext_set=fio.img_ext_set)
        image_paths += seq_image_paths

    for path in tqdm(image_paths):
        (tempdir, tempname, tempext) = fio.get_filename_components(path)
        name = tempname
        img = cv2.imread(path)
        masks = mask_generator.generate(img)

        mask_list = []
        for m in masks:
            m_score = torch.from_numpy(m['segmentation']).float()[None, None, :, :].to('cuda')
            m_score = torch.nn.functional.interpolate(m_score, size=(200,200) , mode='bilinear', align_corners=False).squeeze()
            m_score[m_score >= 0.5] = 1
            m_score[m_score != 1] = 0
            mask_list.append(m_score)
        masks = torch.stack(mask_list, dim=0)

        torch.save(masks, os.path.join(OUTPUT_DIR, name+'.pt'))