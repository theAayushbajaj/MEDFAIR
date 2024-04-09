#!/bin/bash
#SBATCH --mail-user=aayush.bajaj@umontreal.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=dataset_preparation
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=127000M
#SBATCH --time=0-00:45
#SBATCH --account=rrg-ebrahimi

module load opencv/4.5.1 python/3.10.2

SOURCEDIR=/home/aayushb/projects/def-ebrahimi/aayushb/MEDFAIR

source ~/scratch/compute/bin/activate

python $SOURCEDIR/notebooks/HAM1000-preprocess.py