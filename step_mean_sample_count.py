#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
import glob
# from pathlib import Path

# with open('target_path.txt', mode='r', encoding='utf-8') as f:
#     target_path = f.readline().replace('\n', '')
with open('.env', mode='r', encoding='utf-8') as f:
    target_path = "EXPERIMENT/" + f.readline().replace('\n', '')
print("TARGET : {}".format(target_path)) #test

# Path(target_path  + "/" + "steps_mean").mkdir(exist_ok=True)

print("\n")

log_file_list = glob.glob(target_path + "/*.log")
print("!!! log file !!!")
for name in log_file_list:
    print(name.split("/")[-1])

print("\n")

# log_file_name = "eval5000002.log"
log_file_name = str(input("Input a log file name : "))

df = pd.read_csv(target_path + "/" + log_file_name, header=None)

df = df.rename(columns={0:"log", 1:"success"})

get_ind = df[df.iloc[:,0].str.contains("evaluation: FloorPlan\d{1,3}\D",regex=True)].index

get_steps = lambda x : int(re.split("[ ]", x)[4])

print("\n")

# Failure on the way
df_fa_way = []
for ind in get_ind:
    floor_name = df.loc[ind, "log"].split(" ")[1]
    df_by_floor = df.loc[ind-250:ind]
    df_get_step = df_by_floor[df_by_floor["success"].str.contains("False",na=False)].query('not log.str.contains("300")')
    df_get_step.insert(0,"floor_name",floor_name)
    df_fa_way.append(df_get_step)
df_fa_way = pd.concat(df_fa_way).reset_index(drop=True)
df_fa_way["step"] = pd.DataFrame(map(get_steps, df_fa_way["log"]))

get_num = lambda y : int(y.replace("FloorPlan", ""))
df_fa_way_calc = df_fa_way.groupby("floor_name").agg({"step":"mean", "success":"count"}).reset_index(drop=False)
df_fa_way_calc["floor_num"] = pd.DataFrame(map(get_num, df_fa_way_calc["floor_name"]))
df_fa_way_calc_output = df_fa_way_calc.sort_values("floor_num").drop("floor_num", axis=1).rename(columns={"step":"step_mean","success":"sample_count"})
print("{} : \n{}".format("'Failure on the way'", df_fa_way_calc_output))
print("\n")
# df_fa_way_calc_output.to_csv(target_path  + "/" + "steps_mean" + "/" + log_file_name.replace(".log", "_ep_fail_on_the_way.csv"), index=False)


# Failure at max(300steps)
df_fa_max = []
for ind in get_ind:
    floor_name = df.loc[ind, "log"].split(" ")[1]
    df_by_floor = df.loc[ind-250:ind]
    df_get_step = df_by_floor[df_by_floor["success"].str.contains("False",na=False)].query('log.str.contains("300")')
    df_get_step.insert(0,"floor_name",floor_name)
    df_fa_max.append(df_get_step)
df_fa_max = pd.concat(df_fa_max).reset_index(drop=True)
df_fa_max["step"] = pd.DataFrame(map(get_steps, df_fa_max["log"]))

get_num = lambda y : int(y.replace("FloorPlan", ""))
df_fa_max_calc = df_fa_max.groupby("floor_name").agg({"step":"mean", "success":"count"}).reset_index(drop=False)
df_fa_max_calc["floor_num"] = pd.DataFrame(map(get_num, df_fa_max_calc["floor_name"]))
df_fa_max_calc_output = df_fa_max_calc.sort_values("floor_num").drop("floor_num", axis=1).rename(columns={"step":"step_mean","success":"sample_count"})
print("{} : \n{}".format("'Failure at max(300steps)'", df_fa_max_calc_output))
print("\n")
#  df_fa_max_calc_output.to_csv(target_path  + "/" + "steps_mean" + "/" + log_file_name.replace(".log", "_ep_fail_max_300.csv"), index=False)


# Success
df_suc = []
for ind in get_ind:
    floor_name = df.loc[ind, "log"].split(" ")[1]
    df_by_floor = df.loc[ind-250:ind]
    df_get_step = df_by_floor[df_by_floor["success"].str.contains("True",na=False)]
    df_get_step.insert(0,"floor_name",floor_name)
    df_suc.append(df_get_step)
df_suc = pd.concat(df_suc).reset_index(drop=True)
df_suc["step"] = pd.DataFrame(map(get_steps, df_suc["log"]))

get_num = lambda y : int(y.replace("FloorPlan", ""))
df_suc_calc = df_suc.groupby("floor_name").agg({"step":"mean", "success":"count"}).reset_index(drop=False)
df_suc_calc["floor_num"] = pd.DataFrame(map(get_num, df_suc_calc["floor_name"]))
df_suc_calc_output = df_suc_calc.sort_values("floor_num").drop("floor_num", axis=1).rename(columns={"step":"step_mean","success":"sample_count"})
print("{} : \n{}".format("'Success'", df_suc_calc_output))
print("\n")
# df_suc_calc_output.to_csv(target_path  + "/" + "steps_mean" + "/" + log_file_name.replace(".log", "_ep_success.csv"), index=False)
