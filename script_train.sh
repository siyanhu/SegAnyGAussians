python train_scene.py \
    -s /home/siyanhu/Gits/SegAnyGAussians/data/sub2 \
    -m /home/siyanhu/Gits/SegAnyGAussians/data/sub2/output_samtgs \
    --feature_dim 32 && \
python train_contrastive_feature.py \
    -s /home/siyanhu/Gits/SegAnyGAussians/data/sub2 \
    -m /home/siyanhu/Gits/SegAnyGAussians/data/sub2/output_samtgs \
    --feature_dim 32 && \
python train_scene.py \
    -s /home/siyanhu/Gits/SegAnyGAussians/data/sub3 \
    -m /home/siyanhu/Gits/SegAnyGAussians/data/sub3/output_samtgs \
    --feature_dim 32 && \
python train_contrastive_feature.py \
    -s /home/siyanhu/Gits/SegAnyGAussians/data/sub3 \
    -m /home/siyanhu/Gits/SegAnyGAussians/data/sub3/output_samtgs \
    --feature_dim 32 && \
python train_scene.py \
    -s /home/siyanhu/Gits/SegAnyGAussians/data/sub4 \
    -m /home/siyanhu/Gits/SegAnyGAussians/data/sub4/output_samtgs \
    --feature_dim 32 && \
python train_contrastive_feature.py \
    -s /home/siyanhu/Gits/SegAnyGAussians/data/sub4 \
    -m /home/siyanhu/Gits/SegAnyGAussians/data/sub4/output_samtgs \
    --feature_dim 32 && \
python train_scene.py \
    -s /home/siyanhu/Gits/SegAnyGAussians/data/sub5 \
    -m /home/siyanhu/Gits/SegAnyGAussians/data/sub5/output_samtgs \
    --feature_dim 32 && \
python train_contrastive_feature.py \
    -s /home/siyanhu/Gits/SegAnyGAussians/data/sub5 \
    -m /home/siyanhu/Gits/SegAnyGAussians/data/sub5/output_samtgs \
    --feature_dim 32 && \
python train_scene.py \
    -s /home/siyanhu/Gits/SegAnyGAussians/data/sub6 \
    -m /home/siyanhu/Gits/SegAnyGAussians/data/sub6/output_samtgs \
    --feature_dim 32 && \
python train_contrastive_feature.py \
    -s /home/siyanhu/Gits/SegAnyGAussians/data/sub6 \
    -m /home/siyanhu/Gits/SegAnyGAussians/data/sub6/output_samtgs \
    --feature_dim 32