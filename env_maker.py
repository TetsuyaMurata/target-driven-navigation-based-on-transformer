import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--env', default='grid_memory')

args = parser.parse_args()

os.makedirs("EXPERIMENT/" + args.env, exist_ok=True)

with open('.env', mode='w', encoding='utf-8') as f:
    f.write(str(args.env))