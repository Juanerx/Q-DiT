#!/bin/bash

QUANT_FLAGS="--wbits 4 --abits 8 --w_sym --a_sym \
            --act_group_size 128 --weight_group_size 128 \
            --quant_method max \
            --calib_data_path ../cali_data/cali_data_256.pth"

SAMPLE_FLAGS="--batch-size 16 --num-fid-samples 10000 --num-sampling-steps 50 --cfg-scale 1.5 --image-size 256 --seed 0"

export PYTHONUNBUFFERED=1

python -u quant_main.py $QUANT_FLAGS $SAMPLE_FLAGS