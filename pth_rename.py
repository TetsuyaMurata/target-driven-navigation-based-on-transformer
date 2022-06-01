#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import shutil
import glob
import os
import re
from pathlib import Path

# change below directory to the area your pth files are
#os.chdir('/home/dl-box/target-driven-navigation-based-on-transformer/model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/4hist/61/concat/checkpoints/tmp/ok_train_test')
os.chdir('/home/dl-box/target-driven-navigation-based-on-transformer/model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/4hist/61/concat/checkpoints/grid_memory')

Path("Rap_type").mkdir(exist_ok=True) # you can get the Raphael's step pth files in this folder

target_pth_list = [s for s in glob.glob("*.pth")]

df_target = pd.DataFrame(target_pth_list)


for i in range(len(df_target)):
    get_num = lambda x : re.split("[.]", str(x))[0]
    origin_pth_num = int(pd.DataFrame(map(get_num, df_target.iloc[i])).iloc[0].item())
    shutil.copy2(str(origin_pth_num) + ".pth", 'Rap_type/{}.pth'.format(origin_pth_num*5))
