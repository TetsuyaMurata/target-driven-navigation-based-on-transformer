#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import re
import os
import glob

with open('target_path.txt', mode='r', encoding='utf-8') as f:
    target_path = f.readline().replace('\n', '')
os.chdir(target_path)
print("TARGET : {}".format(target_path)) #test

print(glob.glob('*'))

print("\n")

# target_folder = input(str("Which folders should be targeted?:"))
target_folder = "."

os.chdir(target_folder)

read_log_list = glob.glob('*.log')

### All FloorPlan mean function
def calc_success_mean_all_func(df):
    # success_mean
    df_success_mean = df[df.iloc[:,0].str.contains("^episode success:",regex=True)]
    df_success_mean = df_success_mean.reset_index(drop=True)
    df_success_mean[df_success_mean.columns] = df_success_mean[df_success_mean.columns].astype(str)
    func = lambda x : re.split("[ %]", str(x))[2]
    df_success_mean_list = list(map(func, df_success_mean.iloc[:,0]))
    df_success_mean_for_concat = pd.DataFrame(df_success_mean_list)
    success_mean_result_for_concat_all = "{:.2f}".format(df_success_mean_for_concat.astype(float).mean(skipna=True).item())
    if success_mean_result_for_concat_all == "nan":
        success_mean_result_for_concat_all = "{:.2f}".format(0.00)
    return success_mean_result_for_concat_all

def calc_success_over5_mean_all_func(df):
    # success_mean
    df_success_over5_mean = df[df.iloc[:,0].str.contains("^episode > 5 success:",regex=True)]
    df_success_over5_mean = df_success_over5_mean.reset_index(drop=True)
    df_success_over5_mean[df_success_over5_mean.columns] = df_success_over5_mean[df_success_over5_mean.columns].astype(str)
    func = lambda x : re.split("[ %]", str(x))[4]
    df_success_over5_mean_list = list(map(func, df_success_over5_mean.iloc[:,0]))
    df_success_over5_mean_for_concat = pd.DataFrame(df_success_over5_mean_list).rename(columns={0:'value'}).astype('float')
    df_success_over5_mean_for_concat = df_success_over5_mean_for_concat.query('value > 0', engine='python')
    success_over5_mean_result_for_concat_all = "{:.2f}".format(df_success_over5_mean_for_concat.astype(float).mean(skipna=True).item())
    if success_over5_mean_result_for_concat_all == "nan":
        success_over5_mean_result_for_concat_all = "{:.2f}".format(0.00)
    return success_over5_mean_result_for_concat_all

def calc_spl_mean_all_func(df):
    # spl_mean
    df_spl_mean = df[df.iloc[:,0].str.contains("^episode SPL:",regex=True)]
    df_spl_mean = df_spl_mean.reset_index(drop=True)
    df_spl_mean[df_spl_mean.columns] = df_spl_mean[df_spl_mean.columns].astype(str)
    func = lambda x : re.split("[ ]", str(x))[2]
    df_spl_mean_list = list(map(func, df_spl_mean.iloc[:,0]))
    df_spl_mean_for_concat = pd.DataFrame(df_spl_mean_list)
    spl_mean_result_for_concat_all = "{:.2f}".format(100*df_spl_mean_for_concat.astype(float).mean(skipna=True).item())
    if spl_mean_result_for_concat_all == "nan":
        spl_mean_result_for_concat_all = "{:.2f}".format(0.00)
    return spl_mean_result_for_concat_all

def calc_spl_over5_mean_all_func(df):
    # spl_mean
    df_spl_over5_mean = df[df.iloc[:,0].str.contains("^episode SPL > 5:",regex=True)]
    df_spl_over5_mean = df_spl_over5_mean.reset_index(drop=True)
    df_spl_over5_mean[df_spl_over5_mean.columns] = df_spl_over5_mean[df_spl_over5_mean.columns].astype(str)
    func = lambda x : re.split("[ ]", str(x))[4]
    df_spl_over5_mean_list = list(map(func, df_spl_over5_mean.iloc[:,0]))
    df_spl_over5_mean_for_concat = pd.DataFrame(df_spl_over5_mean_list).rename(columns={0:'value'}).astype('float')
    df_spl_over5_mean_for_concat = df_spl_over5_mean_for_concat.query('value > 0', engine='python')
    spl_over5_mean_result_for_concat_all = "{:.2f}".format(100*df_spl_over5_mean_for_concat.astype(float).mean(skipna=True).item())
    if spl_over5_mean_result_for_concat_all == "nan":
        spl_over5_mean_result_for_concat_all = "{:.2f}".format(0.00)
    return spl_over5_mean_result_for_concat_all

def calc_step_mean_all_func(df):
    # step_mean
    df_step_mean = df[df.iloc[:,0].str.contains("^mean episode length:",regex=True)]
    df_step_mean = df_step_mean.reset_index(drop=True)
    df_step_mean[df_step_mean.columns] = df_step_mean[df_step_mean.columns].astype(str)
    func = lambda x : re.split("[ ]", str(x))[3]
    df_step_mean_list = list(map(func, df_step_mean.iloc[:,0]))
    df_step_mean_for_concat = pd.DataFrame(df_step_mean_list)
    step_mean_result_for_concat_all = "{:.2f}".format(df_step_mean_for_concat.astype(float).mean(skipna=True).item())
    if step_mean_result_for_concat_all == "nan":
        step_mean_result_for_concat_all = "{:.2f}".format(0.00)
    return step_mean_result_for_concat_all

### All FloorPlan mean function by Floor Category
for file_name in read_log_list:
    SCENE_TASKS_NG = []
    num_result = []
    df_merge_result = []
    SCENE_TASKS = ["KITCHEN", "LIVING_ROOM", "BEDROOM", "BATHROOM"]
    try:
        df = pd.read_csv(file_name) 
        # ALL scene mean value for concat
        df_merge_mean_all_floor = pd.DataFrame(["ALL",
                                                calc_success_mean_all_func(df),
                                                calc_success_over5_mean_all_func(df),
                                                calc_spl_mean_all_func(df),
                                                calc_spl_over5_mean_all_func(df),
                                                calc_step_mean_all_func(df)]).T
        df_merge_mean_all_floor.columns = ["category", "success_mean", "success(>5)_mean", "spl_mean", "spl(>5)_mean", "step_mean"]

        try:
            print(file_name)
            for i in range(len(SCENE_TASKS)):
                try:
                    if i == 0:
                        df_grouoby_FP = df[df.iloc[:,0].str.contains("evaluation: FloorPlan\d{1,2}\D",regex=True)]
                    elif i == 1:
                        df_grouoby_FP = df[df.iloc[:,0].str.contains("evaluation: FloorPlan2\d{2}\D",regex=True)]
                    elif i == 2:
                        df_grouoby_FP= df[df.iloc[:,0].str.contains("evaluation: FloorPlan3\d{2}\D",regex=True)]
                    elif i == 3:
                        df_grouoby_FP = df[df.iloc[:,0].str.contains("evaluation: FloorPlan4\d{2}\D",regex=True)]

                    # Average value for each FloorPlan
                    df_merge = []
                    for i in df_grouoby_FP.index.values:
                        df_for_concat = df.iloc[i:i + 13, :]
                        df_merge.append(df_for_concat)
                    df_merge = pd.concat(df_merge)

                    # success_mean
                    df_success_mean = df_merge[df_merge.iloc[:,0].str.contains("^episode success",regex=True)]
                    df_success_mean = df_success_mean.reset_index(drop=True)
                    df_success_mean[df_success_mean.columns] = df_success_mean[df_success_mean.columns].astype(str)
                    func = lambda x : re.split("[ %]", str(x))[2]
                    df_success_mean_list = list(map(func, df_success_mean.iloc[:,0]))
                    df_success_mean_for_concat = pd.DataFrame(df_success_mean_list)
                    success_mean_result_for_concat = "{:.2f}".format(df_success_mean_for_concat.astype(float).mean(skipna=True).item())
                    if success_mean_result_for_concat == "nan":
                        success_mean_result_for_concat = "{:.2f}".format(0.00)

                    # success_mean (> 5)
                    df_success_over5_mean = df_merge[df_merge.iloc[:,0].str.contains("^episode > 5 success:",regex=True)]
                    df_success_over5_mean = df_success_over5_mean.reset_index(drop=True)
                    df_success_over5_mean[df_success_over5_mean.columns] = df_success_over5_mean[df_success_over5_mean.columns].astype(str)
                    func = lambda x : re.split("[ %]", str(x))[4]
                    df_success_over5_mean_list = list(map(func, df_success_over5_mean.iloc[:,0]))
                    df_success_over5_mean_for_concat = pd.DataFrame(df_success_over5_mean_list).rename(columns={0:'value'}).astype('float')
                    df_success_over5_mean_for_concat = df_success_over5_mean_for_concat.query('value > 0', engine='python')
                    success_over5_mean_result_for_concat = "{:.2f}".format(df_success_over5_mean_for_concat.astype(float).mean(skipna=True).item())
                    if success_over5_mean_result_for_concat == "nan":
                        success_over5_mean_result_for_concat = "{:.2f}".format(0.00)

                    # spl_mean
                    df_spl_mean = df_merge[df_merge.iloc[:,0].str.contains("^episode SPL:",regex=True)]
                    df_spl_mean = df_spl_mean.reset_index(drop=True)
                    df_spl_mean[df_spl_mean.columns] = df_spl_mean[df_spl_mean.columns].astype(str)
                    func = lambda x : re.split("[ ]", str(x))[2]
                    df_spl_mean_list = list(map(func, df_spl_mean.iloc[:,0]))
                    df_spl_mean_for_concat = pd.DataFrame(df_spl_mean_list)
                    spl_mean_result_for_concat = "{:.2f}".format(100*df_spl_mean_for_concat.astype(float).mean(skipna=True).item())
                    if spl_mean_result_for_concat == "nan":
                        spl_mean_result_for_concat = "{:.2f}".format(0.00)

                    # spl_mean (> 5)
                    df_spl_over5_mean = df_merge[df_merge.iloc[:,0].str.contains("^episode SPL > 5:",regex=True)]
                    df_spl_over5_mean = df_spl_over5_mean.reset_index(drop=True)
                    df_spl_over5_mean[df_spl_over5_mean.columns] = df_spl_over5_mean[df_spl_over5_mean.columns].astype(str)
                    func = lambda x : re.split("[ ]", str(x))[4]
                    df_spl_over5_mean_list = list(map(func, df_spl_over5_mean.iloc[:,0]))
                    df_spl_over5_mean_for_concat = pd.DataFrame(df_spl_over5_mean_list).rename(columns={0:'value'}).astype('float')
                    df_spl_over5_mean_for_concat = df_spl_over5_mean_for_concat.query('value > 0', engine='python')
                    spl_over5_mean_result_for_concat = "{:.2f}".format(100*df_spl_over5_mean_for_concat.astype(float).mean(skipna=True).item())
                    if spl_over5_mean_result_for_concat == "nan":
                        spl_over5_mean_result_for_concat = "{:.2f}".format(0.00)
                   
                    # step_mean
                    df_step_mean = df_merge[df_merge.iloc[:,0].str.contains("^mean episode length:",regex=True)]
                    df_step_mean = df_step_mean.reset_index(drop=True)
                    df_step_mean[df_step_mean.columns] = df_step_mean[df_step_mean.columns].astype(str)
                    func = lambda x : re.split("[ ]", str(x))[3]
                    df_step_mean_list = list(map(func, df_step_mean.iloc[:,0]))
                    df_step_mean_for_concat = pd.DataFrame(df_step_mean_list)
                    step_mean_result_for_concat = "{:.2f}".format(df_step_mean_for_concat.astype(float).mean(skipna=True).item())
                    # step_mean_result_for_concat = "{:.2f}".format(df_step_mean_for_concat.astype(float).mean().item())
                    if step_mean_result_for_concat == "nan":
                        step_mean_result_for_concat = "{:.2f}".format(0.00)

                    num_result.append(success_mean_result_for_concat)
                    num_result.append(success_over5_mean_result_for_concat)
                    num_result.append(spl_mean_result_for_concat)
                    num_result.append(spl_over5_mean_result_for_concat)
                    num_result.append(step_mean_result_for_concat)

                except:
                    print("-----> No Data In " + str(SCENE_TASKS[i]))
                    SCENE_TASKS_NG.append(SCENE_TASKS[i])  
                    continue

            df_merge_result = pd.DataFrame(np.array(num_result).reshape(-1, 5))
            df_merge_result.columns = ["success_mean", "success(>5)_mean", "spl_mean","spl(>5)_mean","step_mean"]
            
            for delete in SCENE_TASKS_NG:
                SCENE_TASKS.remove(delete)

            df_merge_result.loc[:,"category"] = SCENE_TASKS
            df_merge_result = df_merge_result.loc[:, ["category", "success_mean", "success(>5)_mean", "spl_mean", "spl(>5)_mean", "step_mean"]]
            df_merge_result = pd.concat([df_merge_result, df_merge_mean_all_floor]) # Add all scene score mean
            df_merge_result.to_csv("category_score_mean_{}.csv".format(file_name.replace(".log","")),index=False)
            
        except:
            print("ERROR In Log Data " + str(file_name))  
            continue
    
    except:
        print("Can't READ " + str(file_name))
        continue
