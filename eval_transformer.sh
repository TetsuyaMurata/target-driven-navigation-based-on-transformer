#!/bin/bash
source /etc/profile.d/modules.sh
#export SINGULARITY_BINDPATH=/groups/gac50631
#module load singularitypro
#module load cuda/10.2/10.2.89
#module load cudnn/8.0/8.0.5
cd /home/dl-box/target-driven-navigation-based-on-transformer
#source activate env_ai
python memory_size.py #memory_size can be changed
singularity exec --nv /home/dl-box/target-driven-navigation-based-on-transformer/cenotate_transformer.simg python -u eval.py #main
