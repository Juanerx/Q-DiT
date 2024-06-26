#!/bin/bash

module purge
module load anaconda/2021.11 compilers/cuda/11.6 cudnn/8.4.0.27_cuda11.x compilers/gcc/9.3.0 nccl/2.17.1-1_cuda11.6
source activate qdit

QUANT_FLAGS="--wbits 4 --abits 8 \
            --act_group_size 128 --weight_group_size 128 \
            --quant_method max --w_clip_ratio 1 --a_clip_ratio 1 --use_gptq
            --calib_data_path ../cali_data/cali_data_4000.pth"

SAMPLE_FLAGS="--batch-size 16 --num-fid-samples 10000 --num-sampling-steps 50 --cfg-scale 0 --image-size 256 --seed 0"

# export CUDA_VISIBLE_DEVICES="$1"

export PYTHONUNBUFFERED=1

python -u quant_main.py $QUANT_FLAGS $SAMPLE_FLAGS