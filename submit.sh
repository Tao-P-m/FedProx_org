#!/bin/bash
#SBATCH -J test
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:4

module load anaconda3
module load cuda11.2/toolkit/11.2.2
module load cudnn8.1-cuda11.2/8.1.1.33

source activate torch
export PYTHONPATH=$PYTHONPATH:./
echo PYTHONPATH
# drop_percent=0 0.5 0.9 ; mu=NA 0 1
python3  -u main.py --dataset=$1 --optimizer='fedprox'  \
            --learning_rate=0.01 --num_rounds=200 --clients_per_round=10 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --drop_percent=$2 \
            --model='mclr' \
            --mu=$3