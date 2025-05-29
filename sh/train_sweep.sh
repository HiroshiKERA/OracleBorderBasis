#!/bin/bash

# p=7, n=3
CUDA_VISIBLE_DEVICES=0 nohup bash -c "exec -a skim_p7_n3_k1 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p7_n3_k1" > results/train/logs/skim_p7_n3_k1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash -c "exec -a skim_p7_n3_k3 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p7_n3_k3" > results/train/logs/skim_p7_n3_k3.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash -c "exec -a skim_p7_n3_k5 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p7_n3_k5" > results/train/logs/skim_p7_n3_k5.log 2>&1 &

# p=7, n=4 
CUDA_VISIBLE_DEVICES=3 nohup bash -c "exec -a skim_p7_n4_k1 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p7_n4_k1" > results/train/logs/skim_p7_n4_k1.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup bash -c "exec -a skim_p7_n4_k3 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p7_n4_k3" > results/train/logs/skim_p7_n4_k3.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup bash -c "exec -a skim_p7_n4_k5 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p7_n4_k5" > results/train/logs/skim_p7_n4_k5.log 2>&1 &

# # p=7, n=5
CUDA_VISIBLE_DEVICES=6 nohup bash -c "exec -a skim_p7_n5_k1 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p7_n5_k1" > results/train/logs/skim_p7_n5_k1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash -c "exec -a skim_p7_n5_k3 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p7_n5_k3" > results/train/logs/skim_p7_n5_k3.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash -c "exec -a skim_p7_n5_k5 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p7_n5_k5" > results/train/logs/skim_p7_n5_k5.log 2>&1 &


# p=31, n=3
CUDA_VISIBLE_DEVICES=2 nohup bash -c "exec -a skim_p31_n3_k1 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p31_n3_k1" > results/train/logs/skim_p31_n3_k1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash -c "exec -a skim_p31_n3_k3 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p31_n3_k3" > results/train/logs/skim_p31_n3_k3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup bash -c "exec -a skim_p31_n3_k5 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p31_n3_k5" > results/train/logs/skim_p31_n3_k5.log 2>&1 &

# p=31, n=4
CUDA_VISIBLE_DEVICES=5 nohup bash -c "exec -a skim_p31_n4_k1 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p31_n4_k1" > results/train/logs/skim_p31_n4_k1.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup bash -c "exec -a skim_p31_n4_k3 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p31_n4_k3" > results/train/logs/skim_p31_n4_k3.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash -c "exec -a skim_p31_n4_k5 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p31_n4_k5" > results/train/logs/skim_p31_n4_k5.log 2>&1 &

# # p=31, n=5
CUDA_VISIBLE_DEVICES=1 nohup bash -c "exec -a skim_p31_n5_k1 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p31_n5_k1" > results/train/logs/skim_p31_n5_k1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash -c "exec -a skim_p31_n5_k3 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p31_n5_k3" > results/train/logs/skim_p31_n5_k3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash -c "exec -a skim_p31_n5_k5 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p31_n5_k5" > results/train/logs/skim_p31_n5_k5.log 2>&1 &


# p=127, n=3
CUDA_VISIBLE_DEVICES=4 nohup bash -c "exec -a skim_p127_n3_k1 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p127_n3_k1" > results/train/logs/skim_p127_n3_k1.log 2>&1 &    
CUDA_VISIBLE_DEVICES=5 nohup bash -c "exec -a skim_p127_n3_k3 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p127_n3_k3" > results/train/logs/skim_p127_n3_k3.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup bash -c "exec -a skim_p127_n3_k5 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p127_n3_k5" > results/train/logs/skim_p127_n3_k5.log 2>&1 &

# # p=127, n=4
CUDA_VISIBLE_DEVICES=0 nohup bash -c "exec -a skim_p127_n4_k1 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p127_n4_k1" > results/train/logs/skim_p127_n4_k1.log 2>&1 & 
CUDA_VISIBLE_DEVICES=1 nohup bash -c "exec -a skim_p127_n4_k3 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p127_n4_k3" > results/train/logs/skim_p127_n4_k3.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash -c "exec -a skim_p127_n4_k5 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p127_n4_k5" > results/train/logs/skim_p127_n4_k5.log 2>&1 &


# p=127, n=5
CUDA_VISIBLE_DEVICES=3 nohup bash -c "exec -a skim_p127_n5_k1 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p127_n5_k1" > results/train/logs/skim_p127_n5_k1.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup bash -c "exec -a skim_p127_n5_k3 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p127_n5_k3" > results/train/logs/skim_p127_n5_k3.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup bash -c "exec -a skim_p127_n5_k5 python -m scripts.train.train --config config/experiments/sweep_expansion.yaml --experiment skim_p127_n5_k5" > results/train/logs/skim_p127_n5_k5.log 2>&1 &
