#!/usr/bin/env python
import torch
import argparse
import multiprocessing as mp

from agent.training import Training
from agent.utils import populate_config
import json #add

                                                                                                                                                                                                         
with open('.env', mode='r', encoding='utf-8') as f:
    target_path = "EXPERIMENT/" + f.readline().replace('\n', '')
print("TARGET : {}".format(target_path.replace("EXPERIMENT/", "")))

if __name__ == '__main__':
    torch.set_num_threads(1)
    mp.set_start_method('spawn')
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Deep reactive agent.')
    parser.add_argument('--entropy_beta', type=float, default=0.01,
                        help='entropy beta (default: 0.01)')

    parser.add_argument('--restore', action='store_true', help='restore from checkpoint')
    parser.add_argument('--grad_norm', type = float, default=40.0,
        help='gradient norm clip (default: 40.0)')

    #parser.add_argument('--h5_file_path', type = str, default='/app/data/{scene}.h5')
    #parser.add_argument('--h5_file_path', type = str, default='./data/{scene}.h5')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/img_Sum/action/cb4/128d/dropout/3layer/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/3layer/word2vec_action/count/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/word2vec/25M/25cm/4hist/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/word2vec/25M/8hist/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/without_grid/dropout/action/2layer/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/resnet/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/25cm/32hist/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/no_marge/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/concat/32hist/checkpoint-{checkpoint}.pth')
    # parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/4hist/61/concat/checkpoint-{checkpoint}.pth') # origin 20220421
    # parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/4hist/61/concat/checkpoint-{checkpoint}.pth') # add 20220421
    parser.add_argument('--checkpoint_path', type = str, default=target_path + '/' + 'checkpoint-{checkpoint}.pth') # add 20220421 important
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/no_trans/32hist/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/only_cur_obs/4hist/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/without_grid/2layer/16hist/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/without_grid/dropout/action/trans/count/32hist/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/without_grid/dropout/action/trans/count/2layer/checkpoint-{checkpoint}.pth')
    #parser.add_argument('--checkpoint_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/word2vec_action_posi/checkpoint-{checkpoint}.pth')

    parser.add_argument('--learning_rate', type = float, default= 0.0007001643593729748)
    #parser.add_argument('--learning_rate', type = float, default= 0.0001001643593729748)
    #parser.add_argument('--learning_rate', type = float, default= 0.001001643593729748)
    #parser.add_argument('--learning_rate', type = float, default= 0.0005001643593729748)
    #parser.add_argument('--learning_rate', type = float, default= 0.007001643593729748)
    
    parser.add_argument('--rmsp_alpha', type = float, default = 0.99,
        help='decay parameter for RMSProp optimizer (default: 0.99)')
    parser.add_argument('--rmsp_epsilon', type = float, default = 0.1,
        help='epsilon parameter for RMSProp optimizer (default: 0.1)')

    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/img_Sum/action/cb4/128d/dropout/3layer/param.json')
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/word2vec/25M/8hist/param.json')
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/word2vec/25M/25cm/4hist/param.json')
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/resnet/param.json')
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/25cm/32hist/param.json')
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/only_cur_obs/4hist/param.json')
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/concat/32hist/param.json')
    # parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/4hist/61/concat/param.json') # origin 20220421
    parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default=target_path + '/' + 'param.json') # add 20220421
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/no_trans/32hist/param.json')
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/without_grid/dropout/action/trans/count/32hist/param.json')
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/without_grid/2layer/16hist/param.json')
    
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/without_grid/dropout/action/2layer/param.json')
    #parser.add_argument('--exp', type=str,help='Experiment parameters.json file', default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/without_grid/dropout/action/trans/count/2layer/param.json')

    #parser.add_argument('--csv_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/img_Sum/action/cb4/128d/dropout/3layer/samplewriter.csv')
    #parser.add_argument('--csv_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/word2vec/25M/8hist/samplewriter.csv')
    #parser.add_argument('--csv_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/word2vec/25M/25cm/4hist/samplewriter.csv')
    #parser.add_argument('--csv_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/without_grid/dropout/action/trans/count/32hist/samplewriter.csv')
    #parser.add_argument('--csv_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/only_cur_obs/4hist/samplewriter.csv')
    #parser.add_argument('--csv_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/resnet/samplewriter.csv')
    #parser.add_argument('--csv_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/25cm/32hist/samplewriter.csv')
    #parser.add_argument('--csv_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/no_trans/32hist/samplewriter.csv')
    #parser.add_argument('--csv_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/concat/32hist/samplewriter.csv')
    # parser.add_argument('--csv_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/4hist/61/concat/samplewriter.csv') # origin 20220421
    parser.add_argument('--csv_path', type = str, default=target_path + '/' + 'samplewriter.csv') # add 20220421
    #parser.add_argument('--csv_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/without_grid/2layer/16hist/samplewriter.csv')
    #parser.add_argument('--csv_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/without_grid/dropout/action/2layer/samplewriter.csv')
    #parser.add_argument('--csv_path', type = str, default='./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/without_grid/dropout/action/trans/count/2layer/samplewriter.csv')

    args = vars(parser.parse_args())
    args = populate_config(args)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    torch.manual_seed(args['seed'])

    # add    
    json_open = open(target_path + "/"+ "param.json", "r")
    json_load = json.load(json_open)

    device = json_load['train_param']['cuda']
    SSL = json_load['SSL']
    NGPU = json_load['NGPU']
    NThreads = json_load['train_param']['num_thread']
    method = json_load['method']
    print("device : {}, SSL : {}, NGPU : {}, NThreads : {}, method : {}".format(str(device), str(SSL), str(NGPU), str(NThreads), str(method))) #test
    

    if json_load['restore']=="restore":
        t = Training.load_checkpoint(device, SSL, NGPU, NThreads, method, args)
        print("!!!!!!!!!! (load_checkpoint) !!!!!!!!!!\n"*50) #add
    else:
        #t = Training(device, SSL, NGPU, NThreads, method, args)
        t = Training(args)
        print("########## (not load_checkpoint) ##########\n"*50) #add

    t.run()



