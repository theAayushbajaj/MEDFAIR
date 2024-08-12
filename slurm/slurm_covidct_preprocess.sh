#!/bin/bash
#SBATCH --job-name=covidct_preprocess
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=127000M
#SBATCH --time=0-1:00:00
#SBATCH --account=rrg-ebrahimi

module load python/3.11.5 scipy-stack opencv/4.9.0 httpproxy

source ~/scratch/compute/bin/activate

echo "Copying dataset to tmpdir"
cp -r /home/aayushb/projects/def-ebrahimi/aayushb/medical_bias/datasets/COVID_CT_MD/COVID_CT_MD.zip $SLURM_TMPDIR/
unzip $SLURM_TMPDIR/COVID_CT_MD.zip -d $SLURM_TMPDIR/
echo "Dataset copied to tmpdir"

python /home/aayushb/projects/def-ebrahimi/aayushb/medical_bias/MEDFAIR/notebooks/CovidCT.py \
    --data_dir $SLURM_TMPDIR/COVID_CT_MD \

echo "Archiving COVID_CT_MD directory"
tar -czvf $SLURM_TMPDIR/COVID_CT_MD.tar.gz $SLURM_TMPDIR/COVID_CT_MD
echo "Copying archived directory to home directory"
cp $SLURM_TMPDIR/COVID_CT_MD.tar.gz /home/aayushb/projects/def-ebrahimi/aayushb/medical_bias/datasets/COVID_CT_MD