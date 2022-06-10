#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=20:00:00
#$ -j y
#$ -m abe

source /etc/profile.d/modules.sh #abci

module load cuda/10.2/10.2.89 #abci
module load cudnn/8.0/8.0.5 #abci
cd Code/target-driven-navigation-based-on-transformer

source .vna/bin/activate
python memory_size.py #memory_size can be changed
#singularity exec --nv cenotate_transformer.simg python -u train.py #abci
python -u train.py #local