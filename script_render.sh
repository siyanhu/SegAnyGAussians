python render.py \
    -m /home/siyanhu/Gits/SegAnyGAussians/data/sub1/output_samtgs \
    --precomputed_mask /home/siyanhu/Gits/SegAnyGAussians/data/sub1/output_samtgs/segmentation/final_mask.pt \
    --target scene  \
    --segment

python render.py \
    -m /home/siyanhu/Gits/SegAnyGAussians/data/sub1/output_samtgs \
    --precomputed_mask /home/siyanhu/Gits/SegAnyGAussians/data/sub1/output_samtgs/segmentation/final_mask.pt \
    --target contrastive_feature  \
    --segment