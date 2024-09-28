import os
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser

from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)

if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM feature extracting params")
    
    parser.add_argument("--image_root", default='./data/360_v2/garden/', type=str)
    parser.add_argument("--sam_checkpoint_path", default="./dependencies/sam_ckpt/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--image_dir", default='./data/360_v2/garden/images', type=str)

    args = parser.parse_args()
    
    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)
    
    IMAGE_DIR = os.path.join(args.image_root, 'images')
    assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
    OUTPUT_DIR = os.path.join(args.image_root, 'features')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    IMAEG_DIR_REAL = args.image_dir
    
    print("Extracting features...")
    for path in tqdm(os.listdir(IMAEG_DIR_REAL)):
        name = path.split('.')[0]
        img = cv2.imread(os.path.join(IMAEG_DIR_REAL, path))
        img = cv2.resize(img,dsize=(1024,1024),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        predictor.set_image(img)
        features = predictor.features
        torch.save(features, os.path.join(OUTPUT_DIR, name+'.pt'))