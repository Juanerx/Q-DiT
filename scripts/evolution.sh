#!/bin/bash

module purge
module load anaconda/2021.11 compilers/cuda/11.6 cudnn/8.4.0.27_cuda11.x compilers/gcc/9.3.0 nccl/2.17.1-1_cuda11.6
source activate qdit

QUANT_FLAGS="--wbits 4 --abits 8 \
            --act_group_size 128 --weight_group_size 128 \
            --quant_method max"

SAMPLE_FLAGS="--batch-size 16 --num-fid-samples 800 --seed 1234  --cfg-scale 0 --image-size 256 \
                --ref_batch ../evaluations/reference_batch/VIRTUAL_imagenet256_labeled.npz"

EVO_FLAGS="--max_epochs 10 --select_num 10 --population_num 50 \
            --crossover_num 15 --mutation_num 25 --m_prob 0.1 --constraint 128 --alpha 0.5"

export PYTHONUNBUFFERED=1

python evolution.py $QUANT_FLAGS $SAMPLE_FLAGS $EVO_FLAGS