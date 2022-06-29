#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import re
import os
import glob

with open('.env', mode='r', encoding='utf-8') as f:
    target_path = "EXPERIMENT/" + f.readline().replace('\n', '')
os.chdir(target_path)
print("TARGET : {}".format(target_path)) #test

print(glob.glob('*'))

print("\n")

target_folder = "."

print("\n")

os.chdir(target_folder)

# AVERAGE TARGET LIST
success_mean_kitchen_ave_list = []
success_over5_mean_kitchen_ave_list = []
spl_mean_kitchen_ave_list = []
spl_over5_mean_kitchen_ave_list = []
step_mean_kitchen_ave_list = []

success_mean_living_ave_list = []
success_over5_mean_living_ave_list = []
spl_mean_living_ave_list = []
spl_over5_mean_living_ave_list = []
step_mean_living_ave_list = []

success_mean_bedroom_ave_list = []
success_over5_mean_bedroom_ave_list = []
spl_mean_bedroom_ave_list = []
spl_over5_mean_bedroom_ave_list = []
step_mean_bedroom_ave_list = []

success_mean_bathroom_ave_list = []
success_over5_mean_bathroom_ave_list = []
spl_mean_bathroom_ave_list = []
spl_over5_mean_bathroom_ave_list = []
step_mean_bathroom_ave_list = []

success_mean_all_ave_list = []
success_over5_mean_all_ave_list = []
spl_mean_all_ave_list = []
spl_over5_mean_all_ave_list = []
step_mean_all_ave_list = []

print("!!!READ FILES!!!")
read_csv_list = [s for s in glob.glob('*.csv') if re.findall("^category_score_mean_.*.csv$", s)]
for file_name in read_csv_list:
    try:
        print(file_name)
        df = pd.read_csv(file_name)
        target_calc = ['KITCHEN', 'LIVING_ROOM', 'BEDROOM', 'BATHROOM', 'ALL']
        for i in range(len(target_calc)):
            try:
                if i == 0:
                    success_mean_kitchen_ave = df.query('category=="KITCHEN"')["success_mean"].item()
                    success_over5_mean_kitchen_ave = df.query('category=="KITCHEN"')["success(>5)_mean"].item()
                    spl_mean_kitchen_ave = df.query('category=="KITCHEN"')["spl_mean"].item()
                    spl_over5_mean_kitchen_ave = df.query('category=="KITCHEN"')["spl(>5)_mean"].item()
                    step_mean_kitchen_ave = df.query('category=="KITCHEN"')["step_mean"].item()
                    success_mean_kitchen_ave_list.append(success_mean_kitchen_ave)
                    success_over5_mean_kitchen_ave_list.append(success_over5_mean_kitchen_ave)
                    spl_mean_kitchen_ave_list.append(spl_mean_kitchen_ave)
                    spl_over5_mean_kitchen_ave_list.append(spl_over5_mean_kitchen_ave)
                    step_mean_kitchen_ave_list.append(step_mean_kitchen_ave)

                elif i == 1:
                    success_mean_living_ave = df.query('category=="LIVING_ROOM"')["success_mean"].item()
                    success_over5_mean_living_ave = df.query('category=="LIVING_ROOM"')["success(>5)_mean"].item()
                    spl_mean_living_ave = df.query('category=="LIVING_ROOM"')["spl_mean"].item()
                    spl_over5_mean_living_ave = df.query('category=="LIVING_ROOM"')["spl(>5)_mean"].item()
                    step_mean_living_ave = df.query('category=="LIVING_ROOM"')["step_mean"].item()
                    success_mean_living_ave_list.append(success_mean_living_ave)
                    success_over5_mean_living_ave_list.append(success_over5_mean_living_ave)
                    spl_mean_living_ave_list.append(spl_mean_living_ave)
                    spl_over5_mean_living_ave_list.append(spl_over5_mean_living_ave)
                    step_mean_living_ave_list.append(step_mean_living_ave)

                elif i == 2:
                    success_mean_bedroom_ave = df.query('category=="BEDROOM"')["success_mean"].item()
                    success_over5_mean_bedroom_ave = df.query('category=="BEDROOM"')["success(>5)_mean"].item()
                    spl_mean_bedroom_ave = df.query('category=="BEDROOM"')["spl_mean"].item()
                    spl_over5_mean_bedroom_ave = df.query('category=="BEDROOM"')["spl(>5)_mean"].item()
                    step_mean_bedroom_ave = df.query('category=="BEDROOM"')["step_mean"].item()
                    success_mean_bedroom_ave_list.append(success_mean_bedroom_ave)
                    success_over5_mean_bedroom_ave_list.append(success_over5_mean_bedroom_ave)
                    spl_mean_bedroom_ave_list.append(spl_mean_bedroom_ave)
                    spl_over5_mean_bedroom_ave_list.append(spl_over5_mean_bedroom_ave)
                    step_mean_bedroom_ave_list.append(step_mean_bedroom_ave)

                elif i == 3:
                    success_mean_bathroom_ave = df.query('category=="BATHROOM"')["success_mean"].item()
                    success_over5_mean_bathroom_ave = df.query('category=="BATHROOM"')["success(>5)_mean"].item()
                    spl_mean_bathroom_ave = df.query('category=="BATHROOM"')["spl_mean"].item()
                    spl_over5_mean_bathroom_ave = df.query('category=="BATHROOM"')["spl(>5)_mean"].item()
                    step_mean_bathroom_ave = df.query('category=="BATHROOM"')["step_mean"].item()
                    success_mean_bathroom_ave_list.append(success_mean_bathroom_ave)
                    success_over5_mean_bathroom_ave_list.append(success_over5_mean_bathroom_ave)
                    spl_mean_bathroom_ave_list.append(spl_mean_bathroom_ave)
                    spl_over5_mean_bathroom_ave_list.append(spl_over5_mean_bathroom_ave)
                    step_mean_bathroom_ave_list.append(step_mean_bathroom_ave) 

                elif i == 4:
                    success_mean_all_ave = df.query('category=="ALL"')["success_mean"].item()
                    success_over5_mean_all_ave = df.query('category=="ALL"')["success(>5)_mean"].item()
                    spl_mean_all_ave = df.query('category=="ALL"')["spl_mean"].item()
                    spl_over5_mean_all_ave = df.query('category=="ALL"')["spl(>5)_mean"].item()
                    step_mean_all_ave = df.query('category=="ALL"')["step_mean"].item()
                    success_mean_all_ave_list.append(success_mean_all_ave)
                    success_over5_mean_all_ave_list.append(success_over5_mean_all_ave)
                    spl_mean_all_ave_list.append(spl_mean_all_ave)
                    spl_over5_mean_all_ave_list.append(spl_over5_mean_all_ave)
                    step_mean_all_ave_list.append(step_mean_all_ave)

            except:
                print("-----> Data Is Missing In " + target_calc[i])
                continue

    except:
        print("!!!ERROR!!! " + str(file_name))

# Exist over 0
success_mean_kitchen_ave_list = [i for i in success_mean_kitchen_ave_list if i > 0]
success_over5_mean_kitchen_ave_list = [i for i in success_over5_mean_kitchen_ave_list if i > 0]
spl_mean_kitchen_ave_list = [i for i in spl_mean_kitchen_ave_list if i > 0]
spl_over5_mean_kitchen_ave_list = [i for i in spl_over5_mean_kitchen_ave_list if i > 0]
step_mean_kitchen_ave_list = [i for i in step_mean_kitchen_ave_list if i > 0]

success_mean_living_ave_list = [i for i in success_mean_living_ave_list if i > 0]
success_over5_mean_living_ave_list = [i for i in success_over5_mean_living_ave_list if i > 0]
spl_mean_living_ave_list = [i for i in spl_mean_living_ave_list if i > 0]
spl_over5_mean_living_ave_list = [i for i in spl_over5_mean_living_ave_list if i > 0]
step_mean_living_ave_list = [i for i in step_mean_living_ave_list if i > 0]

success_mean_bedroom_ave_list = [i for i in success_mean_bedroom_ave_list if i > 0]
success_over5_mean_bedroom_ave_list = [i for i in success_over5_mean_bedroom_ave_list if i > 0]
spl_mean_bedroom_ave_list = [i for i in spl_mean_bedroom_ave_list if i > 0]
spl_over5_mean_bedroom_ave_list = [i for i in spl_over5_mean_bedroom_ave_list if i > 0]
step_mean_bedroom_ave_list = [i for i in step_mean_bedroom_ave_list if i > 0]

success_mean_bathroom_ave_list = [i for i in success_mean_bathroom_ave_list if i > 0]
success_over5_mean_bathroom_ave_list = [i for i in success_over5_mean_bathroom_ave_list if i > 0]
spl_mean_bathroom_ave_list = [i for i in spl_mean_bathroom_ave_list if i > 0]
spl_over5_mean_bathroom_ave_list = [i for i in spl_over5_mean_bathroom_ave_list if i > 0]
step_mean_bathroom_ave_list = [i for i in step_mean_bathroom_ave_list if i > 0]

success_mean_all_ave_list = [i for i in success_mean_all_ave_list if i > 0]
success_over5_mean_all_ave_list = [i for i in success_over5_mean_all_ave_list if i > 0]
spl_mean_all_ave_list = [i for i in spl_mean_all_ave_list if i > 0]
spl_over5_mean_all_ave_list = [i for i in spl_over5_mean_all_ave_list if i > 0]
step_mean_all_ave_list = [i for i in step_mean_all_ave_list if i > 0]

# average calculation
success_mean_ave_kitchen = round(np.nanmean(success_mean_kitchen_ave_list), 2)
success_over5_mean_ave_kitchen = round(np.nanmean(success_over5_mean_kitchen_ave_list), 2)
spl_mean_ave_kitchen = round(np.nanmean(spl_mean_kitchen_ave_list), 2)
spl_over5_mean_ave_kitchen = round(np.nanmean(spl_over5_mean_kitchen_ave_list), 2)
step_mean_ave_kitchen = round(np.nanmean(step_mean_kitchen_ave_list), 2)
success_mean_ave_living = round(np.nanmean(success_mean_living_ave_list), 2)
success_over5_mean_ave_living = round(np.nanmean(success_over5_mean_living_ave_list), 2)
spl_mean_ave_living = round(np.nanmean(spl_mean_living_ave_list), 2)
spl_over5_mean_ave_living = round(np.nanmean(spl_over5_mean_living_ave_list), 2)
step_mean_ave_living = round(np.nanmean(step_mean_living_ave_list), 2)
success_mean_ave_bedroom = round(np.nanmean(success_mean_bedroom_ave_list), 2)
success_over5_mean_ave_bedroom = round(np.nanmean(success_over5_mean_bedroom_ave_list), 2)
spl_mean_ave_bedroom = round(np.nanmean(spl_mean_bedroom_ave_list), 2)
spl_over5_mean_ave_bedroom = round(np.nanmean(spl_over5_mean_bedroom_ave_list), 2)
step_mean_ave_bedroom = round(np.nanmean(step_mean_bedroom_ave_list), 2)
success_mean_ave_bathroom = round(np.nanmean(success_mean_bathroom_ave_list), 2)
success_over5_mean_ave_bathroom = round(np.nanmean(success_over5_mean_bathroom_ave_list), 2)
spl_mean_ave_bathroom = round(np.nanmean(spl_mean_bathroom_ave_list), 2)
spl_over5_mean_ave_bathroom = round(np.nanmean(spl_over5_mean_bathroom_ave_list), 2)
step_mean_ave_bathroom = round(np.nanmean(step_mean_bathroom_ave_list), 2)
success_mean_ave_all = round(np.nanmean(success_mean_all_ave_list), 2)
success_over5_mean_ave_all = round(np.nanmean(success_over5_mean_all_ave_list), 2)
spl_mean_ave_all = round(np.nanmean(spl_mean_all_ave_list), 2)
spl_over5_mean_ave_all = round(np.nanmean(spl_over5_mean_all_ave_list), 2)
step_mean_ave_all = round(np.nanmean(step_mean_all_ave_list), 2)

print("\n")

# AVE SCORE BY SEAT CATEGORY
print("### CALCULATION RESULTS ###")
print("IN KITCHEN, success_mean_ave : {}".format(success_mean_ave_kitchen))
print("IN KITCHEN, success(>5)_mean_ave : {}".format(success_over5_mean_ave_kitchen))
print("IN KITCHEN, spl_mean_ave : {}".format(spl_mean_ave_kitchen))
print("IN KITCHEN, spl(>5)_mean_ave : {}".format(spl_over5_mean_ave_kitchen))
print("IN KITCHEN, step_mean_ave : {}".format(step_mean_ave_kitchen))
print("\t")
print("IN LIVING_ROOM, success_mean_ave : {}".format(success_mean_ave_living))
print("IN LIVING_ROOM, success(>5)_mean_ave : {}".format(success_over5_mean_ave_living))
print("IN LIVING_ROOM, spl_mean_ave : {}".format(spl_mean_ave_living))
print("IN LIVING_ROOM, spl(>5)_mean_ave : {}".format(spl_over5_mean_ave_living))
print("IN LIVING_ROOM, step_mean_ave : {}".format(step_mean_ave_living))
print("\t")
print("IN BEDROOM, success_mean_ave : {}".format(success_mean_ave_bedroom))
print("IN BEDROOM, success(>5)_mean_ave : {}".format(success_over5_mean_ave_bedroom))
print("IN BEDROOM, spl_mean_ave : {}".format(spl_mean_ave_bedroom))
print("IN BEDROOM, spl(>5)_mean_ave : {}".format(spl_over5_mean_ave_bedroom))
print("IN BEDROOM, step_mean_ave : {}".format(step_mean_ave_bedroom))
print("\t")
print("IN BATHROOM, success_mean_ave : {}".format(success_mean_ave_bathroom))
print("IN BATHROOM, success(>5)_mean_ave : {}".format(success_over5_mean_ave_bathroom))
print("IN BATHROOM, spl_mean_ave : {}".format(spl_mean_ave_bathroom))
print("IN BATHROOM, spl(>5)_mean_ave : {}".format(spl_over5_mean_ave_bathroom))
print("IN BATHROOM, step_mean_ave : {}".format(step_mean_ave_bathroom))
print("\t")
print("IN ALL, success_mean_ave : {}".format(success_mean_ave_all))
print("IN ALL, success(>5)_mean_ave : {}".format(success_over5_mean_ave_all))
print("IN ALL, spl_mean_ave : {}".format(spl_mean_ave_all))
print("IN ALL, spl(>5)_mean_ave : {}".format(spl_over5_mean_ave_all))
print("IN ALL, step_mean_ave : {}".format(step_mean_ave_all))
print("\t")
