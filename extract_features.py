import os
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser

import root_file_io as fio

from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)

if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM feature extracting params")
    
    parser.add_argument("--image_root", default='./data/360_v2/garden/', type=str)
    parser.add_argument("--sam_checkpoint_path", default="./dependencies/sam_ckpt/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)

    args = parser.parse_args()
    
    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)
    
    IMAGE_DIR = os.path.join(args.image_root, 'images')
    assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
    OUTPUT_DIR = os.path.join(args.image_root, 'features')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Extracting features...")
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
        img = cv2.resize(img,dsize=(1024,1024),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        predictor.set_image(img)
        features = predictor.features
        torch.save(features, os.path.join(OUTPUT_DIR, name+'.pt'))