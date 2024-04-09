#!/bin/bash
#SBATCH --mail-user=aayush.bajaj@umontreal.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=HAM10k
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=22G
#SBATCH --time=0-2:00:00
#SBATCH --account=rrg-ebrahimi

module load python/3.11.5 scipy-stack opencv/4.9.0 cuda

source ~/scratch/medfair/bin/activate

python main.py --experiment baseline --dataset_name HAM10000 \
     --total_epochs 20 --sensitive_name Sex --batch_size 1024 \
     --sens_classes 2 --output_dim 1 --num_classes 1

echo "done"