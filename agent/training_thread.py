from agent.network import SceneSpecificNetwork, SharedNetwork, ActorCriticLoss
from agent.environment import Environment, THORDiscreteEnvironment
import torch.nn as nn
from typing import Dict, Collection
import signal
import random
import torch
import h5py
from agent.replay import ReplayMemory, Sample
from collections import namedtuple
import torch.multiprocessing as mp
import torch.nn.functional as F
from agent.method.similarity_grid import SimilarityGrid
import numpy as np
import logging
from multiprocessing import Condition
import matplotlib.pyplot as plt
import csv
import os
from torchvision import transforms
import math
import json

with open('.env', mode='r', encoding='utf-8') as f:
    target_path = "EXPERIMENT/" + f.readline().replace('\n', '')

json_open = open(target_path + "/"+ "param.json", "r")
json_load = json.load(json_open)

memory_size_read = json_load['memory']
print("(tr_thread) memory_size : {}".format(str(memory_size_read))) #test

TrainingSample = namedtuple('TrainingSample', ('state', 'policy', 'value', 'action_taken', 'goal', 'R', 'temporary_difference'))

"""
A neural memory module inspired by the Scene Memory Transformer paper.
"""

import typing
from collections import deque

import numpy as np


"""
The Pytorch implementation of the memory.
"""

def reset_memory(memory, mask):
    assert mask.shape[0] == memory.shape[0], f"Memory sizes don't match"
    #assert len(reset.shape.as_list()) == 0, f"Reset must be scalar"
    new_memory = memory * 0
    #new_mask = mask * 0
    new_mask = torch.ones(new_memory.shape[0])
    return new_memory, new_mask

class TrainingThread(mp.Process):
    def __init__(self,
            id : int,
            network : torch.nn.Module,
            saver,
            device,
            optimizer,
            tasks : list,
            #method,
            #action_size,
            summary_queue: mp.Queue,
            **kwargs):

        super(TrainingThread, self).__init__()

        # Initialize the environment
        self.env = None
        self.init_args = kwargs
        self.saver = saver
        #self.local_backbone_network = SharedNetwork()
        self.id = id
        self.device = device
        self.tasks = tasks
        #self.method = method
        self.scenes = set([scene for (scene, target) in tasks])
        self.master_network = network
        self.optimizer = optimizer
        self.summary_queue = summary_queue
        if torch.cuda.is_available():
            self.gpu = True 
        else:
            self.gpu = False
        self.exit = mp.Event()

    def _sync_network(self):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                state_dict = self.master_network.state_dict()
                self.policy_network.load_state_dict(state_dict)
        else:
            state_dict = self.master_network.state_dict()
            self.policy_network.load_state_dict(state_dict)

    def _ensure_shared_grads(self):
        for param, shared_param in zip(self.policy_network.parameters(), self.master_network.parameters()):
            if shared_param.grad is not None:
                return 
            shared_param._grad = param.grad 
    
    def get_action_space_size(self):
        return len(self.envs[0].actions)

    def _initialize_thread(self):
        torch.set_num_threads(1)
        h5_file_path = self.init_args.get('h5_file_path')
        #print(self.init_args.get('action_size'))
        #self.action_size= self.init_args.get('action_size')

        #self.device= self.init_args.get('device')
        #self.method= self.init_args.get('method')
        self.init_args['h5_file_path'] = lambda scene: h5_file_path.replace('{scene}', scene)
        print("aaaaaaaaaaaaaaaaaa")
        if self.init_args.get('method')=='word2vec_notarget' or self.init_args.get('method')=='Transformer_word2vec_notarget'or self.init_args.get('method')=='Transformer_word2vec_notarget_withposi' or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_posi' or self.init_args.get('method')=="gcn_transformer" or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_concat' or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action' or self.init_args.get('method')=='grid_memory' or self.init_args.get('method')=='Transformer_word2vec_notarget_action'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action_posi'or self.init_args.get('method')=='grid_memory_action':
            self.envs = [THORDiscreteEnvironment(
                                         scene_name=scene,
                                         terminal_state=task,
                                         **self.init_args)
            for (scene, task) in self.tasks]
        else:
            self.envs = [THORDiscreteEnvironment(scene_name = scene,
                                                 terminal_state_id = int(task),
                                                 **self.init_args)
            for (scene,task) in self.tasks]
        print("yyyyyyyyyyyyyyyy")
        self.gamma : float = self.init_args.get('gamma', 0.99)
        self.grad_norm: float = self.init_args.get('grad_norm', 40.0)
        entropy_beta : float = self.init_args.get('entropy_beta', 0.01)
        #self.max_t : int = self.init_args.get('max_t', 5)
        self.max_t : int = self.init_args.get('max_t')
        self.local_t = 0
        #self.action_space_size = self.get_action_space_size()
        self.action_space_size = self.init_args.get('action_size')

        self.criterion = ActorCriticLoss(entropy_beta)
        #self.policy_network = nn.Sequential(SharedNetwork(self.method), SceneSpecificNetwork(self.get_action_space_size(),self.method))
        self.policy_network = nn.Sequential(SharedNetwork(self.init_args.get('method')), SceneSpecificNetwork(self.action_space_size,self.init_args.get('method')))
        self.policy_network = self.policy_network.to(self.device)
        # Initialize the episode
        for idx, _ in enumerate(self.envs):
            self._reset_episode(idx)
        self._sync_network()
        self.add_reward = self.init_args.get('ADDREWARD')

        #####Memory#######
        if self.init_args.get('method') == "Transformer_Concat" or self.init_args.get('method') == "Transformer_Sum" or self.init_args.get('method')=='Transformer_word2vec_notarget'or self.init_args.get('method')=='Transformer_word2vec_notarget_withposi' or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_posi' or self.init_args.get('method')=="gcn_transformer"or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_concat'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action'or self.init_args.get('method')=='grid_memory'or self.init_args.get('method')=='Transformer_word2vec_notarget_action'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action_posi'or self.init_args.get('method')=='grid_memory_action':
            # self.memory_size = 32 #origin
            self.memory_size = memory_size_read #add
            self.embed_size = 2048
            self.batch_size = 1
            self.memory = torch.zeros(self.memory_size, self.embed_size)            
            self.mask = torch.ones(self.memory_size)
            self.theta = 0
            self.positions = [[0,0, self.theta] for i in range(self.memory_size)]
        if self.init_args.get('method') == 'gcn_transformer':
            self.gcn_memory = torch.zeros(self.memory_size, 3, 300, 400)
            self.gcn_mask = torch.ones(self.memory_size)
        if self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action'or self.init_args.get('method')=='Transformer_word2vec_notarget_action'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action_posi'or self.init_args.get('method')=='grid_memory_action':
            #self.act_memory = torch.zeros(self.memory_size, 8)
            #self.act_mask = torch.ones(self.memory_size)
            #self.last_act = torch.zeros(8).to(self.device)
            self.act_memory = torch.zeros(self.memory_size, 128)
            self.act_mask = torch.ones(self.memory_size)
            self.last_act = torch.zeros(128).to(self.device)
        if self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action_posi':
            self.locs = np.zeros((self.memory_size, 4))
            self.rllocs = np.zeros((self.memory_size, 4))
            self.deg = np.zeros((1,1))
            self.nlocs = 0
        if self.init_args.get('method')=='grid_memory'or self.init_args.get('method')=='grid_memory_action':
            self.grid_memory = torch.zeros(self.memory_size, 16, 16)
            self.grid_mask = torch.ones(self.memory_size)
        if self.init_args.get('method') == "word2vec_notarget" or self.init_args.get('method')=='Transformer_word2vec_notarget'or self.init_args.get('method')=='Transformer_word2vec_notarget_withposi' or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_posi'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_concat'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action'or self.init_args.get('method')=='grid_memory':
            self.method_class = SimilarityGrid(self.init_args.get('method'))
        self.saved_actions = []
        if self.add_reward == 'count':
            #self.states = []
            self.states = [[-1] for i in range(self.memory_size)]
            self.prev_states = [[-1] for i in range(self.memory_size)]
            self.actions = [[-1] for i in range(self.memory_size)]


    def _reset_episode(self,idx):
        self.saved_actions = []
        self.episode_reward = 0
        self.episode_length = 0
        #self.episode_max_q = -np.inf
        self.episode_max_q = torch.FloatTensor([-np.inf]).to(self.device)
        self.envs[idx].reset()

    def extract_input(self, env, device):
            if self.init_args.get('method') == 'word2vec_notarget' or self.init_args.get('method')=='Transformer_word2vec_notarget'or self.init_args.get('method')=='Transformer_word2vec_notarget_withposi' or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_posi'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_concat'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action'or self.init_args.get('method')=='grid_memory'or self.init_args.get('method')=='Transformer_word2vec_notarget_action'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action_posi'or self.init_args.get('method')=='grid_memory_action':
                state = {
                    "current": env.render('resnet_features'),
                    "goal": env.render_target('word_features')
                }
                state["object_mask"] = env.render_mask_similarity()
                x_processed = torch.from_numpy(state["current"])
                object_mask = torch.from_numpy(state['object_mask'])

                x_processed = x_processed.to(device)
                object_mask = object_mask.to(device)

                return state, x_processed, object_mask
            elif self.init_args.get('method') == 'gcn_transformer':
                normalize = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225])])
                state = {
                        "current": env.render('resnet_features'),
                        "goal": env.render_target('word_features'),
                        "observation": normalize(env.observation).unsqueeze(0),
                }
                state["object_mask"] = env.render_mask_similarity()
                x_processed = torch.from_numpy(state["current"])
                goal_processed = torch.from_numpy(state["goal"])
                object_mask = torch.from_numpy(state['object_mask'])
                obs = state['observation']

                x_processed = x_processed.to(device)
                goal_processed = goal_processed.to(device)
                object_mask = object_mask.to(device)
                obs = obs.to(device)

                return state, x_processed, goal_processed, object_mask, obs



    def _forward_explore(self,scene,idx):
        global REWARDS,LENGTHES
        # Does the evaluation end naturally?
        is_terminal = False
        terminal_end = False
        self.env = self.envs[idx]
        results = { "policy":[], "value": []}
        rollout_path = {"state": [], "action": [], "rewards": [], "done": []}

        # Plays out one game to end or max_t
        for t in range(self.max_t):
            if self.init_args.get('method') =='word2vec_notarget' or self.init_args.get('method')=='Transformer_word2vec_notarget'or self.init_args.get('method')=='Transformer_word2vec_notarget_withposi' or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_posi'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_concat'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action'or self.init_args.get('method')=='grid_memory'or self.init_args.get('method')=='Transformer_word2vec_notarget_action'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action_posi'or self.init_args.get('method')=='grid_memory_action':
                state, x_processed, object_mask = self.extract_input(self.env, self.device)
            elif self.init_args.get('method')=='gcn_transformer':
                state, x_processed, goal_processed, object_mask, obs = self.extract_input(self.env, self.device)
            else:
                state = { 
                    "current": self.env.render('resnet_features'),
                    "goal": self.env.render_target('resnet_features'),
                }


            if self.init_args.get('method')=='pretrain_action_posi'or self.init_args.get('method')=='Transformer_word2vec_notarget_action_posi':
                home_posi = np.array([self.env.location]) / 5
                #print(self.deg)
                home_posi = np.append(home_posi, self.deg, axis=1)
                home_posi = np.append(home_posi, self.deg, axis=1)
                self.nlocs += 1
                #print(home_posi)
                #print(self.locs.shape)
                self.locs = np.delete(self.locs, -1, axis=0)
                #print(self.locs.shape)
                self.locs = np.append(home_posi, self.locs, axis=0)
                #print(self.locs)
                if self.nlocs < self.memory_size:
                    self.rllocs[0:self.nlocs] = self.locs[0:self.nlocs] - home_posi
                    #print(self.rllocs)
                    for m in range(self.nlocs):
                        self.rllocs[m][2] = np.sin(np.deg2rad(self.rllocs[m][2]))
                        self.rllocs[m][3] = np.cos(np.deg2rad(self.rllocs[m][3]))
                        #print(self.rllocs)
                else:
                    self.rllocs[:] = self.locs[:] - home_posi
                    #print(self.rllocs)
                    for m in range(self.memory_size):
                        self.rllocs[m][2] = np.sin(np.deg2rad(self.rllocs[m][2]))
                        self.rllocs[m][3] = np.cos(np.deg2rad(self.rllocs[m][3]))
                        #print(self.rllocs)
                        #print(self.locs)
                        #print(self.rllocs)
            if self.init_args.get('method') == "Baseline":
                x_processed = torch.from_numpy(state["current"])
                goal_processed = torch.from_numpy(state["goal"])
                x_processed = x_processed.to(self.device)
                goal_processed = goal_processed.to(self.device)
                (policy, value) = self.policy_network((x_processed, goal_processed,))
            elif self.init_args.get('method') == "word2vec_notarget":
                (policy, value) = self.policy_network((x_processed, object_mask,))
                #policy, value, state = self.method_class.forward_policy(self.envs[idx], self.device, self.policy_network)
            elif self.init_args.get('method')=='Transformer_word2vec_notarget'or self.init_args.get('method')=='Transformer_word2vec_notarget_withposi' or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_concat':
                if self.init_args.get('Posi'):
                    (policy, value, memory, mask) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.device, self.positions))
                elif self.init_args.get('Key')=="word2vec":
                    (policy, value, memory, mask,_,_) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.device, self.env.s_target))
                else:
                    (policy, value, memory, mask, _, _) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.device))
                if self.device != "cpu":
                    self.memory = memory.clone().detach().cpu()
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = mask.clone().detach().cpu()
                    self.mask = self.mask.view(self.memory_size)
                else:
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = self.mask.view(self.memory_size)
            elif self.init_args.get('method')=='Transformer_word2vec_notarget_action':
                (policy, value, memory, mask,act_memory, act_mask,_,_) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.device, self.last_act, self.act_memory, self.act_mask))
                if self.device != "cpu":
                    self.memory = memory.clone().detach().cpu()
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = mask.clone().detach().cpu()
                    self.mask = self.mask.view(self.memory_size)
                    self.act_memory = act_memory.clone().detach().cpu()
                    #self.act_memory = self.act_memory.view(self.memory_size, 8)
                    self.act_memory = self.act_memory.view(self.memory_size, 128)
                    self.act_mask = act_mask.clone().detach().cpu()
                    self.act_mask = self.act_mask.view(self.memory_size)
                else:
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = self.mask.view(self.memory_size)
                    #self.act_memory = self.act_memory.view(self.memory_size,8)
                    self.act_memory = self.act_memory.view(self.memory_size,128)
                    self.act_mask = self.act_mask.view(self.memory_size)
            elif self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action_posi':
                (policy, value, memory, mask,act_memory, act_mask,_,_) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.device, self.env.s_target, self.last_act, self.act_memory, self.act_mask, self.rllocs))
                if self.device != "cpu":
                    self.memory = memory.clone().detach().cpu()
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = mask.clone().detach().cpu()
                    self.mask = self.mask.view(self.memory_size)
                    self.act_memory = act_memory.clone().detach().cpu()
                    self.act_memory = self.act_memory.view(self.memory_size, 8)
                    self.act_mask = act_mask.clone().detach().cpu()
                    self.act_mask = self.act_mask.view(self.memory_size)
                else:
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = self.mask.view(self.memory_size)
                    self.act_memory = self.act_memory.view(self.memory_size,8)
                    self.act_mask = self.act_mask.view(self.memory_size)
            elif self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_posi':
                (policy, value, memory, mask,_,_) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.device, self.positions, self.env.s_target))
                if self.device != "cpu":
                    self.memory = memory.clone().detach().cpu()
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = mask.clone().detach().cpu()
                    self.mask = self.mask.view(self.memory_size)
                else:
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = self.mask.view(self.memory_size)
            elif self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action':
                (policy, value, memory, mask,act_memory, act_mask,_,_) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.device, self.env.s_target, self.last_act, self.act_memory, self.act_mask))
                if self.device != "cpu":
                    self.memory = memory.clone().detach().cpu()
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = mask.clone().detach().cpu()
                    self.mask = self.mask.view(self.memory_size)
                    self.act_memory = act_memory.clone().detach().cpu()
                    self.act_memory = self.act_memory.view(self.memory_size, 8)
                    self.act_mask = act_mask.clone().detach().cpu()
                    self.act_mask = self.act_mask.view(self.memory_size)
                else:
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = self.mask.view(self.memory_size)
                    self.act_memory = self.act_memory.view(self.memory_size,8)
                    self.act_mask = self.act_mask.view(self.memory_size)
            elif self.init_args.get('method')=='gcn_transformer':
                (policy, value, memory, mask, gcn_memory, gcn_mask) = self.policy_network((x_processed[:,-1], goal_processed, object_mask, obs, self.memory, self.mask, self.gcn_memory, self.gcn_mask, self.device))
                if self.device != "cpu":
                    self.memory = memory.clone().detach().cpu()
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = mask.clone().detach().cpu()
                    self.mask = self.mask.view(self.memory_size)
                    self.gcn_memory = gcn_memory.clone().detach().cpu()
                    self.gcn_memory = self.gcn_memory.view(self.memory_size, 3, 300, 400)
                    self.gcn_mask = gcn_mask.clone().detach().cpu()
                    self.gcn_mask = self.gcn_mask.view(self.memory_size)
                else:
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = self.mask.view(self.memory_size)
                    self.gcn_memory = self.gcn_memory.view(self.memory_size, 3, 300, 400)
                    self.gcn_mask = self.gcn_mask.view(self.memory_size)
            elif self.init_args.get('method')=='grid_memory':
                (policy, value, memory, mask, grid_memory, grid_mask, _, _) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.env.s_target, self.grid_memory, self.grid_mask, self.device))
                if self.device != "cpu":
                    self.memory = memory.clone().detach().cpu()
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = mask.clone().detach().cpu()
                    self.mask = self.mask.view(self.memory_size)
                    self.grid_memory = grid_memory.clone().detach().cpu()
                    self.grid_memory = self.grid_memory.view(self.memory_size, 16, 16)
                    self.grid_mask = grid_mask.clone().detach().cpu()
                    self.grid_mask = self.grid_mask.view(self.memory_size)
                else:
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = self.mask.view(self.memory_size)
                    self.grid_memory = self.grid_memory.view(self.memory_size, 16,16)
                    self.grid_mask = self.grid_mask.view(self.memory_size)
            elif self.init_args.get('method')=='grid_memory_action':
                (policy, value, memory, mask, grid_memory, grid_mask,act_memory, act_mask, _, _) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.env.s_target, self.grid_memory, self.grid_mask, self.last_act, self.act_memory, self.act_mask, self.device))
                if self.device != "cpu":
                    self.memory = memory.clone().detach().cpu()
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = mask.clone().detach().cpu()
                    self.mask = self.mask.view(self.memory_size)
                    self.grid_memory = grid_memory.clone().detach().cpu()
                    self.grid_memory = self.grid_memory.view(self.memory_size, 16, 16)
                    self.grid_mask = grid_mask.clone().detach().cpu()
                    self.grid_mask = self.grid_mask.view(self.memory_size)
                    self.act_memory = act_memory.clone().detach().cpu()
                    self.act_memory = self.act_memory.view(self.memory_size, 128)
                    self.act_mask = act_mask.clone().detach().cpu()
                    self.act_mask = self.act_mask.view(self.memory_size)
                else:
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = self.mask.view(self.memory_size)
                    self.grid_memory = self.grid_memory.view(self.memory_size, 16,16)
                    self.grid_mask = self.grid_mask.view(self.memory_size)
                    self.act_memory = self.act_memory.view(self.memory_size, 128)
                    self.act_mask = self.act_mask.view(self.memory_size)
            else:
                x_processed = torch.from_numpy(state["current"][:,-1])
                goal_processed = torch.from_numpy(state["goal"][:,-1])
                x_processed = x_processed.to(self.device)
                goal_processed = goal_processed.to(self.device)
                (policy, value, memory, mask) = self.policy_network((x_processed, goal_processed, self.memory, self.mask, self.device))
                if self.device != "cpu":
                    self.memory = memory.clone().detach().cpu()
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = mask.clone().detach().cpu()
                    self.mask = self.mask.view(self.memory_size)
                else:
                    self.memory = self.memory.view(self.memory_size, self.embed_size)
                    self.mask = self.mask.view(self.memory_size)


            # Store raw network output to use in backprop
            results["policy"].append(policy)
            results["value"].append(value)

            with torch.no_grad():
                (_, action,) = policy.max(0)
                action = F.softmax(policy, dim=0).multinomial(1).item()
            policy = policy.data
            value = value.data
            
            if self.add_reward == 'count' and action!= 8:
                del self.prev_states[0]
                self.prev_states.append(self.env.current_state_id)

            # Makes the step in the environment
            self.env.step(action)
            if self.init_args.get('Posi'):
                del self.positions[0]
                if self.env.collided:
                    self.positions.append([0,0, self.theta])
                elif action == 0:
                    self.positions.append([1,0,self.theta])
                elif action == 3:
                    self.positions.append([-1,0, self.theta])
                elif action == 6:
                    self.positions.append([0,1,self.theta])
                elif action == 7:
                    self.positions.append([0,-1,self.theta])
                elif action == 1:
                    self.theta +=1
                    self.positions.append([0,0,self.theta])
                elif action == 2:
                    self.theta-=1
                    self.positions.append([0,0,self.theta])
                else:
                    self.positions.append([0,0,self.theta])

            self.saved_actions.append(action)

            if self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action' or self.init_args.get('method')=='Transformer_word2vec_notarget_action' or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action_posi' or self.init_args.get('method')== 'grid_memory_action':
                if action == 0:
                    self.last_act = torch.tensor(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                elif action == 1:
                    self.last_act = torch.tensor(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                elif action == 2:
                    self.last_act = torch.tensor(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                elif action == 3:
                    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                elif action == 4:
                    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                elif action == 5:
                    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                elif action == 6:
                    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), dtype=torch.float32).to(self.device)
                elif action == 7:
                    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), dtype=torch.float32).to(self.device)
                #if action == 0:
                #    self.last_act = torch.tensor(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                #elif action == 1:
                #    self.last_act = torch.tensor(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                #elif action == 2:
                #    self.last_act = torch.tensor(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                #elif action == 3:
                #    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                #elif action == 4:
                #    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                #elif action == 5:
                #    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                #elif action == 6:
                #    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), dtype=torch.float32).to(self.device)
                #elif action == 7:
                #    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), dtype=torch.float32).to(self.device)
                #if action == 0:
                #    self.last_act = torch.tensor(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                #elif action == 1:
                #    self.last_act = torch.tensor(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                #elif action == 2:
                #    self.last_act = torch.tensor(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                #elif action == 3:
                #    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                #elif action == 4:
                #    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                #elif action == 5:
                #    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                #elif action == 6:
                #    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), dtype=torch.float32).to(self.device)
                #elif action == 7:
                #    self.last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), dtype=torch.float32).to(self.device)

            # ad-hoc reward for navigation
            #reward = 10.0 if is_terminal else -0.01
            #if self.add_reward == 'None':
            #    reward = self.env.reward
            if self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action_posi':
                if action == 1:
                    self.deg += 45.0
                elif action == 2:
                    self.deg -= 45.0
            reward = self.env.reward
            #if self.add_reward == 'count' and action!= 8:
            #    del self.states[0]
            #    self.states.append(self.env.current_state_id)
            #    n = self.states.count(self.env.current_state_id)
            #    reward += -(1.5**(n-5)/100.0) 
            if self.add_reward == 'count' and action!= 8:
                del self.actions[0]
                self.actions.append(action)
                print(self.actions)
                print(self.prev_states)
                print(action)
                print(self.env.current_state_id)
                #LIST = [i for i, x in enumerate(self.prev_states) if x == self.env.current_state_id]
                LIST = [i for i, x in enumerate(self.prev_states) if x == self.prev_states[-1]]
                print("#########")
                print(LIST)
                print("#########")
                CCC = 0 
                for e in LIST:
                    if self.actions[e] == action:
                        CCC +=1
                print(CCC)
                if CCC >= 2:
                    #reward += -0.01
                    reward += -(0.01 * (CCC-1))
                    #reward += -0.05
                    #reward += -0.1
                    #pass
            #if self.add_reward == 'count' and action!= 8:
            #    self.states.append(self.env.current_state_id)
            #    n = self.states.count(self.env.current_state_id)
            #    reward += 0.01 -(math.sqrt(n)/100.0) 
            
            # Receives the game reward
            is_terminal = self.env.is_terminal

            # Max episode length
            #if self.episode_length > 10e3: is_terminal = True #10000
            if self.init_args.get('method') == 'word2vec_notarget' or self.init_args.get('method')=='Transformer_word2vec_notarget'or self.init_args.get('method')=='Transformer_word2vec_notarget_withposi' or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_posi' or self.init_args.get('method')=='gcn_transformer'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_concat'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action' or self.init_args.get('method')=='grid_memory'or self.init_args.get('method')=='Transformer_word2vec_notarget_action'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action_posi'or self.init_args.get('method')=='grid_memory_action':
                if self.episode_length > 200: is_terminal = True    #200
            else:
                if self.episode_length > 5000: is_terminal = True #10000
            # Update episode stats
            self.episode_length += 1
            self.episode_reward += reward
            with torch.no_grad():
                self.episode_max_q = torch.max(self.episode_max_q, torch.max(value))

            # clip reward
            reward = np.clip(reward, -1, 1)

            # Increase local time
            self.local_t += 1

            rollout_path["state"].append(state)
            rollout_path["action"].append(action)
            rollout_path["rewards"].append(reward)
            rollout_path["done"].append(is_terminal)
            if is_terminal:
                (_, task) = self.tasks[idx]
                if self.init_args.get('method') == 'word2vec_notarget' or self.init_args.get('method')=='Transformer_word2vec_notarget' or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_posi' or self.init_args.get('method')=='gcn_transformer'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_concat'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action'or self.init_args.get('method')=='grid_memory'or self.init_args.get('method')=='Transformer_word2vec_notarget_action'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action_posi'or self.init_args.get('method')=='grid_memory_action':
                    scene_log = scene + '-' + \
                        str(task['object'])
                else:
                    scene_log = scene + '-' + \
                        str(task)
                step = self.optimizer.get_global_step() * self.max_t
                # TODO: add logging
                print('playout finished')
                print(f'episode length: {self.episode_length}')
                print(f'episode reward: {self.episode_reward}')
                print(f'episode max_q: {self.episode_max_q}')
                
                hist_action, _ = np.histogram(
                        self.saved_actions, bins=self.action_space_size, density=False)
                self.summary_queue.put(
                        (scene_log + '/actions', hist_action, step))

                    # Send info to logger thread
                self.summary_queue.put(
                        (scene_log + '/episode_length', self.episode_length, step))
                self.summary_queue.put(
                    (scene_log + '/max_q', float(self.episode_max_q.detach().cpu().numpy()[0]), step))
                self.summary_queue.put(
                    (scene_log + '/reward', float(self.episode_reward), step))
                self.summary_queue.put(
                    (scene_log + '/learning_rate', float(self.optimizer.scheduler.get_lr()[0]), step))
                
                if self.id == 0:
                    csv_path = self.init_args.get('csv_path')
                    #csv_path = '/model/Transformer/'+str(self.id)+'samplewriter.csv' 
                    if not os.path.isfile(csv_path):
                        with open(csv_path, "w") as f:   # ファイルを作成
                            f.write(" not found" + '\n')   # 最後の \n は改行コード
                            f.write(" created" + '\n')
                    else:
                        with open(csv_path, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([self.id, self.optimizer.get_global_step(), self.episode_length]) 
                terminal_end = True
                self._reset_episode(idx)
                if self.init_args.get('method') == "Transformer_Sum" or self.init_args.get('method') == "Transformer_Concat" or self.init_args.get('method')=='Transformer_word2vec_notarget'or self.init_args.get('method')=='Transformer_word2vec_notarget_withposi' or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_posi' or self.init_args.get('method')=='gcn_transformer'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_concat'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action'or self.init_args.get('method')=='grid_memory'or self.init_args.get('method')=='Transformer_word2vec_notarget_action'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action_posi'or self.init_args.get('method')=='grid_memory_action':
                    self.memory, self.mask = reset_memory(self.memory, self.mask)
                    self.theta = 0
                    self.positions = [[0,0, self.theta] for i in range(self.memory_size)]
                if self.init_args.get('method')=='gcn_transformer':
                    self.gcn_memory, self.gcn_mask = reset_memory(self.gcn_memory, self.gcn_mask)
                if self.init_args.get('method')=='grid_memory'or self.init_args.get('method')=='grid_memory_action':
                    self.grid_memory, self.grid_mask = reset_memory(self.grid_memory, self.grid_mask)
                if self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action'or self.init_args.get('method')=='Transformer_word2vec_notarget_action'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action_posi'or self.init_args.get('method')=='grid_memory_action':
                    self.act_memory, self.act_mask = reset_memory(self.act_memory, self.act_mask)
                    #self.last_act = torch.zeros(8).to(self.device)
                    self.last_act = torch.zeros(128).to(self.device)
                if self.init_args.get('method')=='Transformer_word2vec_notraget_word2vec_action_posi':
                    self.locs = np.zeros((self.memory_size, 4))
                    self.rllocs = np.zeros((self.memory_size, 4))
                    self.deg = np.zeros((1, 1))
                    self.nlocs = 0
                if self.add_reward == 'count':
                    #self.states = []
                    self.states = [[-1] for i in range(self.memory_size)]
                    self.prev_states = [[-1] for i in range(self.memory_size)]
                    self.actions = [[-1] for i in range(self.memory_size)]
                break

        if terminal_end:
            return 0.0, results, rollout_path, terminal_end
        else:
            if self.init_args.get('method') == "Baseline":
                x_processed = torch.from_numpy(self.env.render('resnet_features'))
                goal_processed = torch.from_numpy(self.env.render_target('resnet_features'))
                x_processed = x_processed.to(self.device)
                goal_processed = goal_processed.to(self.device)
                (_, value) = self.policy_network((x_processed, goal_processed,))
            elif self.init_args.get('method') == "word2vec_notarget":
                state, x_processed, object_mask = self.extract_input(self.env, self.device)
                (policy, value) = self.policy_network((x_processed, object_mask,))
            elif self.init_args.get('method')=='Transformer_word2vec_notarget'or self.init_args.get('method')=='Transformer_word2vec_notarget_withposi' or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec'or self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_concat':
                state, x_processed, object_mask = self.extract_input(self.env, self.device)
                if self.init_args.get('Posi'):
                    (policy, value, memory, mask) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.device, self.positions))
                elif self.init_args.get('Key')=="word2vec":
                    (policy, value, memory, mask, _, _) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.device, self.env.s_target))
                else:
                    (policy, value, memory, mask,_,_) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.device))
            elif self.init_args.get('method')=='Transformer_word2vec_notarget_action':
                state, x_processed, object_mask = self.extract_input(self.env, self.device)
                (policy, value, memory, mask,act_memory,act_mask,_,_) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.device, self.last_act, self.act_memory, self.act_mask))
            elif self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action_posi':
                state, x_processed, object_mask = self.extract_input(self.env, self.device)
                (policy, value, memory, mask,act_memory,act_mask,_,_) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.device, self.env.s_target, self.last_act, self.act_memory, self.act_mask, self.rllocs))
            elif self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_posi':
                state, x_processed, object_mask = self.extract_input(self.env, self.device)
                (policy, value, memory, mask,_,_) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.device, self.positions, self.env.s_target))
            elif self.init_args.get('method')=='Transformer_word2vec_notarget_word2vec_action':
                state, x_processed, object_mask = self.extract_input(self.env, self.device)
                (policy, value, memory, mask,act_memory,act_mask,_,_) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.device, self.env.s_target, self.last_act, self.act_memory, self.act_mask))
            elif self.init_args.get('method')=='gcn_transformer':
                state, x_processed, goal_processed, object_mask, obs = self.extract_input(self.env, self.device)
                (_, value, _, _, _, _) = self.policy_network((x_processed[:,-1], goal_processed, object_mask, obs, self.memory, self.mask, self.gcn_memory, self.gcn_mask, self.device))
            elif self.init_args.get('method')=='grid_memory':
                state, x_processed, object_mask = self.extract_input(self.env, self.device)
                (_, value, _, _, _, _, _, _) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.env.s_target, self.grid_memory, self.grid_mask, self.device))
            elif self.init_args.get('method')=='grid_memory_action':
                state, x_processed, object_mask = self.extract_input(self.env, self.device)
                (_, value, _, _, _, _, _, _, _, _) = self.policy_network((x_processed[:,-1], object_mask, self.memory, self.mask, self.env.s_target, self.grid_memory, self.grid_mask, self.last_act, self.act_memory, self.act_mask, self.device))
            else:
                x_processed = torch.from_numpy(self.env.render('resnet_features')[:,-1])
                goal_processed = torch.from_numpy(self.env.render_target('resnet_features')[:,-1])
                x_processed = x_processed.to(self.device)
                goal_processed = goal_processed.to(self.device)
                (_, value, _, _) = self.policy_network((x_processed, goal_processed, self.memory, self.mask, self.device))
            return value.data.item(), results, rollout_path, terminal_end
 
    def _optimize_path(self, playout_reward: float, results, rollout_path):
        policy_batch = []
        value_batch = []
        action_batch = []
        temporary_difference_batch = []
        playout_reward_batch = []


        for i in reversed(range(len(results["value"]))):
            reward = rollout_path["rewards"][i]
            value = results["value"][i]
            action = rollout_path["action"][i]

            playout_reward = reward + self.gamma * playout_reward
            temporary_difference = playout_reward - value.data.item()

            policy_batch.append(results['policy'][i])
            value_batch.append(results['value'][i])
            action_batch.append(action)
            temporary_difference_batch.append(temporary_difference)
            playout_reward_batch.append(playout_reward)
        
        policy_batch = torch.stack(policy_batch, 0).to(self.device)
        value_batch = torch.stack(value_batch, 0).to(self.device)
        action_batch = torch.from_numpy(np.array(action_batch, dtype=np.int64)).to(self.device)
        temporary_difference_batch = torch.from_numpy(np.array(temporary_difference_batch, dtype=np.float32)).to(self.device)
        playout_reward_batch = torch.from_numpy(np.array(playout_reward_batch, dtype=np.float32)).to(self.device)
        
        # Compute loss
        loss = self.criterion.forward(policy_batch, value_batch, action_batch, temporary_difference_batch, playout_reward_batch)
        loss = loss.sum()

        #Loss = loss.data

        #if self.id == 0:
        #    #csv_path = self.init_args.get('csv_path')
        #    csv_path = './loss/'+str(self.id)+'samplewriter.csv' 
        #    if not os.path.isfile(csv_path):
        #        with open(csv_path, "w") as f:   # ファイルを作成
        #            f.write(" not found" + '\n')   # 最後の \n は改行コード
        #            f.write(" created" + '\n')
        #    else:
        #        with open(csv_path, 'a') as f:
        #            writer = csv.writer(f)
        #            writer.writerow([self.id, self.optimizer.get_global_step(), Loss]) 

        self.optimizer.optimize(loss, 
            self.policy_network, 
            self.master_network,
            self.gpu)

    def run(self, master = None):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(f'Thread {self.id} ready')
        
        # We need to silence all errors on new process
        h5py._errors.silence_errors()
        self._initialize_thread()
                
        if not master is None:
            print(f'Master thread {self.id} started')
        else:
            print(f'Thread {self.id} started')

        try:
            idx = [j for j in range(len(self.tasks))]
            random.shuffle(idx)
            j = 0
            for i in range(len(self.envs)):
                self.envs[i].reset()
            #while True:
            while not self.exit.is_set() and self.optimizer.get_global_step() * self.max_t < self.init_args["total_step"]:
                (scene, target) = self.tasks[idx[j]]
                terminal = False

                while not terminal and not self.exit.is_set() and self.optimizer.get_global_step() * self.max_t < self.init_args["total_step"]:
                #while not terminal:
                    self._sync_network()
                    # Plays some samples
                    playout_reward, results, rollout_path, terminal = self._forward_explore(scene,idx[j])
                    # Train on collected samples
                    self._optimize_path(playout_reward, results, rollout_path)
                    
                    # print(f'Step finished {self.optimizer.get_global_step()}') #origin
                    print(f'Step finished {self.optimizer.get_global_step()*5}') #add
                    # Trigger save or other
                    self.saver.after_optimization(self.id)

                j = j+1
                j = j % len(self.tasks)
            #pass
            self.stop()
            [env.stop() for env in self.envs]
        except Exception as e:
            print(e)
            # TODO: add logging
	    
            raise e
    def stop(self):
        print("Stop initiated")
        self.exit.set()
