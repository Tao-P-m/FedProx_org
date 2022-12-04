#!/usr/bin/env bash
#SBATCH -J tfl_lr1e-2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:4

module load anaconda3
module load cuda11.2/toolkit/11.2.2
module load cudnn8.1-cuda11.2/8.1.1.33

source activate tfl
export PYTHONPATH=$PYTHONPATH:./
echo PYTHONPATH

python3  -u main.py --dataset=$1 --optimizer='fedavg'  \
            --learning_rate=0.0001 --num_rounds=200 --M=10 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --model='mclr' \
            --drop_percent=$2 \