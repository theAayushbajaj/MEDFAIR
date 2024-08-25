#!/bin/bash
#SBATCH --job-name=covidct
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-1:00:00
#SBATCH --account=rrg-ebrahimi

module load python/3.11.5 scipy-stack opencv/4.9.0 cuda httpproxy

source ~/scratch/medfair/bin/activate

# Copy dataset to tmpdir
echo "Copying dataset to tmpdir"
cp -r /home/aayushb/projects/def-ebrahimi/aayushb/medical_bias/datasets/Fitz17k/finalfitz17k.zip $SLURM_TMPDIR/
unzip $SLURM_TMPDIR/finalfitz17k.zip -d $SLURM_TMPDIR/finalfitz17k
echo "Dataset copied to tmpdir"

# Executing the script
echo "Executing the script"
python main.py --experiment baseline \
     --dataset_name Fitz17k \
     --random_seed 0 \
     --total_epochs 20 \
     --sensitive_name skin_type \
     --batch_size 64 \
     --output_dim 1 \
     --num_classes 1 \
     --data_dir $SLURM_TMPDIR/finalfitz17k \
     --backbone cusResNet50 \

echo "done"