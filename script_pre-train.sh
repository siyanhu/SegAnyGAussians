# python extract_features.py \
#     --image_root data/sub3 \
#     --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
#     --sam_arch vit_b && \
# python extract_segment_everything_masks.py \
#     --image_root data/sub3 \
#     --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
#     --sam_arch vit_b && \
# python extract_features.py \
#     --image_root data/sub4 \
#     --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
#     --sam_arch vit_b && \
# python extract_segment_everything_masks.py \
#     --image_root data/sub4 \
#     --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
#     --sam_arch vit_b && \
# python extract_features.py \
#     --image_root data/sub5 \
#     --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
#     --sam_arch vit_b && \
# python extract_segment_everything_masks.py \
#     --image_root data/sub5 \
#     --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
#     --sam_arch vit_b && \
# python extract_features.py \
#     --image_root data/sub6 \
#     --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
#     --sam_arch vit_b && \
# python extract_segment_everything_masks.py \
#     --image_root data/sub6 \
#     --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
#     --sam_arch vit_b

python extract_features.py \
    --image_root /home/siyanhu/Gits/SegAnyGAussians/data/4223_0 \
    --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
    --sam_arch vit_b && \
python extract_segment_everything_masks.py \
    --image_root /home/siyanhu/Gits/SegAnyGAussians/data/4223_0 \
    --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
    --sam_arch vit_b

python extract_features.py \
    --image_root /home/siyanhu/Gits/SegAnyGAussians/data/4223_1 \
    --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
    --sam_arch vit_b && \
python extract_segment_everything_masks.py \
    --image_root /home/siyanhu/Gits/SegAnyGAussians/data/4223_1 \
    --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
    --sam_arch vit_b


python extract_features.py \
    --image_root /home/siyanhu/Gits/SegAnyGAussians/data/red2 \
    --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
    --sam_arch vit_b && \
python extract_segment_everything_masks.py \
    --image_root /home/siyanhu/Gits/SegAnyGAussians/data/red2 \
    --sam_checkpoint_path third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth \
    --sam_arch vit_b