#!/bin/bash 
#SBATCH --job-name=PONG36
#SBATCH --time=48:00:00
##SBATCH --partition=exlusive
#SBATCH --nodes 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=2G

#SBATCH --hint=multithread 
#SBATCH --threads-per-core=2

# CD
cd /tmpdir/delatorr/dev/MAGE_ATARI

# LOAD MODULES
module purge
module load julia/1.9.3
module load intelmpi chdb/1.0

export JULIA_DEPOT_PATH=/tmpdir/delatorr/.julia
#export UTCGP_PYTHON=~/.conda/envs/mage/bin/python
#export OMP_NUM_THREADS=4 #${SLURM_CPU_PER_TASK}
#echo $OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=1

# export UTCGP_CONSTRAINED=yes
# export UTCGP_MIN_INT=-10000
# export UTCGP_MAX_INT=10000
# export UTCGP_MIN_FLOAT=-10000
# export UTCGP_MAX_FLOAT=10000
# export UTCGP_SMALL_ARRAY=100
# export UTCGP_BIG_ARRAY=1000

echo $(placement 1 36 --mode=compact --ascii-art)

# TMP=/tmpdir/delatorr/dev/MAGE_ATARI/pong/
srun $(placement 1 36 --mode=compact) julia --project --threads=36 --startup-file=no scripts/training_loop_me_2d.jl 

# 1> \$TMP/$STDOUT 2> \$TMP/$STDERR
