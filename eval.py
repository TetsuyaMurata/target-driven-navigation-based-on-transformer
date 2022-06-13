#!/usr/bin/env python
import torch
import argparse
import multiprocessing as mp

from agent.evaluation import Evaluation
from agent.utils import populate_config
from pathlib import Path

Path("tmp").mkdir(exist_ok=True)

with open('.target_path', mode='r', encoding='utf-8') as f_target:
    target_path = f_target.readline()
print("!!! target_path !!! : {}".format(target_path))


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Deep reactive agent.')
    parser.add_argument('--h5_file_path', type = str, default='./data/{scene}.h5')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/3layer/word2vec/25M/8hist/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/word2vec/25M/4hist/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/img_Sum/action/cb4/128d/dropout/3layer/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/word2vec_action_posi/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/without_grid/dropout/02/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/only_cur_obs/4hist/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/25cm/32hist/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/forvideo/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/concat/32hist/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/4hist/61/wo_object_memory/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/forvideo/know/checkpoint-{checkpoint}.pth')#origin
    # parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/4hist/61/concat/checkpoints-{checkpoint}.pth')#ADD
    # parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/32hist/61/checkpoints-{checkpoint}.pth')#origin 20220421
    # parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/32hist/61/checkpoints-{checkpoint}.pth')#ADD 20220421
    parser.add_argument('--checkpoint_path', type = str, default=target_path + '/' + 'checkpoints-{checkpoint}.pth')#ADD 20220421
    #parser.add_argument('--csv_file', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/word2vec/25M/4hist/result.csv')
    #parser.add_argument('--csv_file', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/img_Sum/action/cb4/128d/dropout/3layer/result.csv')
    #parser.add_argument('--csv_file', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/without_grid/dropout/02/result.csv')
    #parser.add_argument('--csv_file', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/25cm/32hist/result.csv')
    #parser.add_argument('--csv_file', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/forvideo/result.csv')
    #parser.add_argument('--csv_file', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/concat/32hist/result.csv')
    #parser.add_argument('--csv_file', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/4hist/61/wo_object_memory/result.csv')
    # parser.add_argument('--csv_file', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/forvideo/know/result.csv') #origin
    # parser.add_argument('--csv_file', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/4hist/61/concat/result.csv') #ADD
    # parser.add_argument('--csv_file', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/32hist/61/result.csv') #ADD 20220421
    parser.add_argument('--csv_file', type = str, default=target_path + '/' + 'result.csv') #ADD 20220421
    #parser.add_argument('--csv_file', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/only_cur_obs/4hist/result.csv')

    parser.add_argument('--scenes', dest='test_scenes',action='append', help='Scenes to evaluate on', required=False, default=[], type = str)

    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/word2vec/25M/4hist/param.json')
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/img_Sum/action/cb4/128d/dropout/3layer/param.json')
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/25cm/32hist/param.json')
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/forvideo/param.json')
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/concat/32hist/param.json')
    # parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/forvideo/know/param.json') #origin
    # parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/4hist/61/concat/param.json') #ADD
    # parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/32hist/61/param.json') #ADD 20220421
    parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default=target_path + '/' + 'param.json') #ADD 20220421
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/no_trans/32hist/param.json')
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/only_cur_obs/4hist/param.json')
    
    args = vars(parser.parse_args())
    args = populate_config(args, mode='eval')
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #SSL = False #or True
    #method = "Baseline"
    #method = "Transformer_Concat"

    #t = Evaluation.load_checkpoint(args,device, SSL, method)
    t = Evaluation.load_checkpoint(args)
    t.run()

