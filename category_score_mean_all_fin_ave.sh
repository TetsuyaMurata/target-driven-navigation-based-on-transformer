#!/bin/bash
# module load singularitypro
cd /home/dl-box/target-driven-navigation-based-on-transformer
singularity exec /home/dl-box/visual-navigation-agent-pytorch/success_mean.simg python category_score_mean_all_fin_ave.py
