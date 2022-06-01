#!/bin/bash
#module load singularitypro
cd /home/dl-box/target-driven-navigation-based-on-transformer
singularity exec /home/dl-box/visual-navigation-agent-pytorch/success_mean.simg python step_mean_sample_count.py
