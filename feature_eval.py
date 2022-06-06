#!/usr/bin/env python
import argparse
import multiprocessing as mp
import sys

import torch

from agent.feature_evaluation import FeatureEvaluation
from agent.utils import populate_config

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(
        description='Feature importance evaluation using random shuffle.')

    # Use experiment.json
    parser.add_argument('--exp', '-e', type=str,
                        help='Experiment parameters.json file', required=True)

    args = vars(parser.parse_args())
    args = populate_config(args, mode='eval')

    if args.get('method', None) is None:
        print('ERROR Please choose a method in json file')
        print('- "aop"')
        print('- "word2vec"')
        print('- "word2vec_noconv"')
        print('- "word2vec_nosimi"')
        print('- "target_driven"')
        print('- "random"')

        exit()
    else:
        if not args['method'].startswith('word2vec'):
            print(args['method'], "method cannot be used")
            print("Only word2vec")
            exit()

    t = FeatureEvaluation.load_checkpoints(args)
    t.run()
