import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--memory', default=32)

args = parser.parse_args()

with open(".memory_size", "w", encoding="utf-8") as f:
    f.write(str(args.memory))
