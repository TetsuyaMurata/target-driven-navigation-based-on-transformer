#!/bin/bash
<<<<<<< HEAD
#source /etc/profile.d/modules.sh #abci
#export SINGULARITY_BINDPATH=/groups/gac50631 #abci
#module load singularitypro #abci
#module load cuda/10.2/10.2.89 #abci
#module load cudnn/8.0/8.0.5 #abci
#cd target-driven-navigation-based-on-transformer
python memory_size.py #memory_size can be changed
#singularity exec --nv /home/dl-box/target-driven-navigation-based-on-transformer/cenotate_transformer.simg python -u train.py #abci
=======
python memory_size.py #memory_size can be changed
>>>>>>> 617470c79de4457614c5b4a7e962a9af9dc1fac5
python -u train.py #local
