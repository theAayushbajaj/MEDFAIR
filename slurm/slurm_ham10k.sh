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
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-1:00:00
#SBATCH --account=rrg-ebrahimi

module load python/3.11.5 scipy-stack opencv/4.9.0 cuda httpproxy

source ~/scratch/medfair/bin/activate

# Copy dataset to tmpdir
echo "Copying dataset to tmpdir"
cp -r /home/aayushb/projects/def-ebrahimi/aayushb/datasets/HAM10000/HAM10000.tar.gz $SLURM_TMPDIR/
tar -xzf $SLURM_TMPDIR/HAM10000.tar.gz -C $SLURM_TMPDIR/
echo "Dataset copied to tmpdir"

# Executing the script
echo "Executing the script"
python main.py --experiment baseline \
     --dataset_name HAM10000 \
     --random_seed 0 \
     --total_epochs 20 \
     --sensitive_name Age \
     --batch_size 1024 \
     --sens_classes 4 \
     --output_dim 1 \
     --num_classes 1 \
     --data_dir $SLURM_TMPDIR/HAM10000 \
     --backbone cusDenseNet121

echo "done"