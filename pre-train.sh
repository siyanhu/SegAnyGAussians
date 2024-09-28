# python extract_features.py \
#     --image_root <path to the scene data> \
#     --sam_checkpoint_path <path to the pre-trained SAM model> \
#     --downsample <1/2/4/8>

# python extract_segment_everything_masks.py \
#     --image_root <path to the scene data> \
#     --sam_checkpoint_path <path to the pre-trained SAM model> \
#     --downsample <1/2/4/8>

cd data/sub2/images/ && \
mv seq2/* . && \
rm -rf seq2 && \
cd ../../..

python extract_features.py \
    --image_root data/sub2 \
    --image_dir data/sub2/images/seq2 \
    --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
    --sam_arch vit_b

python extract_segment_everything_masks.py \
    --image_root data/sub2 \
    --image_dir data/sub2/images/seq2 \
    --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
    --sam_arch vit_b \