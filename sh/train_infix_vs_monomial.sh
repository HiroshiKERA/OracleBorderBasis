#!/bin/bash
# CONFIG=expansion
CONFIG=prod

# full training
EXP=base_n=2
CUDA_VISIBLE_DEVICES=5 nohup python -m scripts.train.train \
                    --config config/experiments/$CONFIG.yaml \
                    --experiment $EXP \
                    --max_sequence_length 5000 &

EXP=monomial_n=2
CUDA_VISIBLE_DEVICES=5 nohup python -m scripts.train.train \
                    --config config/experiments/$CONFIG.yaml \
                    --experiment $EXP \
                    --max_sequence_length 5000 &

EXP=base_n=3
CUDA_VISIBLE_DEVICES=5 nohup python -m scripts.train.train \
                    --config config/experiments/$CONFIG.yaml \
                    --experiment $EXP \
                    --max_sequence_length 5000 &

EXP=monomial_n=3
CUDA_VISIBLE_DEVICES=6 nohup python -m scripts.train.train \
                    --config config/experiments/$CONFIG.yaml \
                    --experiment $EXP \
                    --max_sequence_length 5000 &

EXP=base_n=4 
CUDA_VISIBLE_DEVICES=6 nohup python -m scripts.train.train \
                    --config config/experiments/$CONFIG.yaml \
                    --experiment $EXP \
                    --max_sequence_length 5000 &

EXP=monomial_n=4
CUDA_VISIBLE_DEVICES=6 nohup python -m scripts.train.train \
                    --config config/experiments/$CONFIG.yaml \
                    --experiment $EXP \
                    --max_sequence_length 5000 &

EXP=base_n=5
CUDA_VISIBLE_DEVICES=7 nohup python -m scripts.train.train \
                    --config config/experiments/$CONFIG.yaml \
                    --experiment $EXP \
                    --max_sequence_length 5000 &
                    
EXP=monomial_n=5
CUDA_VISIBLE_DEVICES=7 nohup python -m scripts.train.train \
                    --config config/experiments/$CONFIG.yaml \
                    --experiment $EXP \
                    --max_sequence_length 5000 &


