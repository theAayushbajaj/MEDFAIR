#!/bin/bash
#SBATCH --job-name=covidct
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-2:00:00
#SBATCH --account=rrg-ebrahimi

module load python/3.11.5 scipy-stack opencv/4.9.0 cuda httpproxy

source ~/scratch/medfair/bin/activate

# Copy dataset to tmpdir
echo "Copying dataset to tmpdir"
cp -r /home/aayushb/projects/def-ebrahimi/aayushb/medical_bias/datasets/PAPILA/papila.zip $SLURM_TMPDIR/
unzip $SLURM_TMPDIR/papila.zip -d $SLURM_TMPDIR/papila
echo "Dataset copied to tmpdir"


echo "Executing the script"
python /home/aayushb/projects/def-ebrahimi/aayushb/medical_bias/MEDFAIR/main.py --experiment baseline \
     --dataset_name PAPILA \
     --total_epochs 20 \
     --sensitive_name Sex \
     --batch_size 256 \
    --sens_classes 2 \
     --output_dim 1 \
     --num_classes 1 \
     --data_dir $SLURM_TMPDIR/papila \
     --backbone cusResNet18

echo "done"