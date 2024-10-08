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
cp -r /home/aayushb/projects/def-ebrahimi/aayushb/medical_bias/datasets/COVID_CT_MD/COVID_CT_MD.tar.gz $SLURM_TMPDIR/
tar -xzf $SLURM_TMPDIR/COVID_CT_MD.tar.gz -C $SLURM_TMPDIR/
echo "Dataset copied to tmpdir"

# Executing the script
echo "Executing the script"
python main.py --experiment baseline \
     --dataset_name COVID_CT_MD \
     --random_seed 0 \
     --total_epochs 20 \
     --sensitive_name Age \
     --batch_size 1024 \
     --sens_classes 5 \
     --output_dim 1 \
     --num_classes 1 \
     --data_dir $SLURM_TMPDIR/COVID_CT_MD \
     --backbone cusDenseNet121

echo "done"