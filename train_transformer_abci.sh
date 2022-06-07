#!/bin/bash
source /etc/profile.d/modules.sh
export SINGULARITY_BINDPATH=/groups/gac50631
module load singularitypro
module load cuda/10.2/10.2.89
module load cudnn/8.0/8.0.5
python memory_size.py #memory_size can be changed
singularity exec --nv cenotate_transformer.simg python -u train.py