#!/bin/bash
source /etc/profile.d/modules.sh
#export SINGULARITY_BINDPATH=/groups/gac50631/Rui/target-driven-navigation-based-on-transformer
#export SINGULARITY_BINDPATH=/groups/gac50631
#export SINGULARITY_BINDPATH=/groups/gac50631/dataset/visual-navigation-agent-pytorch/h5_files_8action_45degree/data
#export SINGULARITY_BINDPATH=/groups/gac50631/dataset/visual-navigation-agent-pytorch/h5_files_25cm_8action_45degree/data
#module load singularitypro
#module load cuda/10.2/10.2.89
#module load cudnn/8.0/8.0.5
cd /home/dl-box/target-driven-navigation-based-on-transformer
#source activate env_ai
python memory_size.py #memory_size can be changed
singularity exec --nv /home/dl-box/target-driven-navigation-based-on-transformer/cenotate_transformer.simg python -u train.py #origin
#python -u train.py #add
#singularity exec --nv /cenotate_transformer.simg python -u train.py
#singularity exec --nv /home/aac13109bi/visual-navigation-agent-pytorch/visual-navigation-agent-pytorch.simg python -u train.py -e EXPERIMENTS/glove_sim_conv16_softgoal_notarget_40scenes/param.json
