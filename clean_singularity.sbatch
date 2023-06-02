#!/bin/bash
#
## Required settings for Tahoma:
#
#SBATCH -p analysis
#SBATCH -N 1
#SBATCH --gres gpu:2
#
## CHANGE_ME: You must tailor these 5 SBATCH settings:
#
#SBATCH -A emsle60535
#SBATCH -D /tahoma/emsle60535/clean_gpu
#SBATCH --time=24:00:00
#SBATCH --mail-user=yyang45@kent.edu

# Optional settings:
#
#SBATCH --mail-type END,BEGIN,FAIL
#SBATCH -o ./clean-%j.stdout
#SBATCH -e ./clean-%j.stderr
#SBATCH --job-name=clean_training

# CHANGE_ME: set ALPHAFOLD_DIR same as "SBATCH -D" value
export CLEAN_DIR=/tahoma/emsle60535/clean_gpu
export TMPDIR=/big_scratch/${SLURM_JOB_ID}

# CHANGE_ME: Tailor this to be input FASTA file:
# export FASTA_PATHS="O00141.fasta"
# VALUES=({7001..8000})
# files=`ls ${FASTAS_DIR}/*.fasta | head -n $VALUES | tail -n 1`
#files=`ls ${FASTAS_DIR}/*.fasta | head -n $SLURM_ARRAY_TASK_ID | tail -n 1`

# CHANGE_ME: Tailor this to be the activation script for your python virtual environment:
source /tahoma/emsle60535/singularity-venv/bin/activate

mkdir $TMPDIR
mkdir $TMPDIR/model
mkdir $TMPDIR/distance_map
echo $TMPDIR

cp $CLEAN_DIR/singularity/*.csv $TMPDIR/

### Check values of some environment variables
echo SLURM_JOB_GPUS=$SLURM_JOB_GPUS
echo CLEAN_DIR=$CLEAN_DIR

export XLA_PYTHON_CLIENT_MEM_FRACTION=10
export TF_FORCE_UNIFIED_MEMORY=1

# Run AlphaFold; default is to use GPUs, i.e. "--use_gpu" can be omitted.
# module load python
python3 ${CLEAN_DIR}/singularity/run_singularity.py \
    --use_gpu \
    --learning_rate=5e-4 \
    --epoch=2000 \
    --model_name=split10_triplet \
    --training_data=split10 \
    --hidden_dim=512 \
    --out_dim=128 

echo INFO: AlphaFold returned $?

### Copy Alphafold output back to directory where "sbatch" command was issued.
mkdir $SLURM_SUBMIT_DIR/Output-$SLURM_JOB_ID
cp -R $TMPDIR $SLURM_SUBMIT_DIR/Output-$SLURM_JOB_ID