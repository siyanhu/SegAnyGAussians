python train_scene.py \
    -s /home/siyanhu/Gits/SegAnyGAussians/data/sub1 \
    -m /home/siyanhu/Gits/SegAnyGAussians/data/sub1/output_samtgs \
    --feature_dim 32


python train_contrastive_feature.py \
    -s /home/siyanhu/Gits/SegAnyGAussians/data/sub1 \
    -m /home/siyanhu/Gits/SegAnyGAussians/data/sub1/output \
    --feature_dim 32
