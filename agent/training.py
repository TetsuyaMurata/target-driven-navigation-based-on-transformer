from agent.network import SharedNetwork, SceneSpecificNetwork
from agent.training_thread import TrainingThread
from agent.optim import SharedRMSprop
from agent.adam import SharedAdam
from typing import Collection, List
import torch.nn as nn
import torch.multiprocessing as mp 
import logging
import sys
import torch
import os
import threading
from contextlib import suppress
import re
import time
from agent.environment import Environment, THORDiscreteEnvironment
from agent.summary_thread import SummaryThread
import json #add
# TOTAL_PROCESSED_FRAMES = 25 * 10**6 # 25 million frames
#TOTAL_PROCESSED_FRAMES = 50 * 10**6 # 50 million frames
#TOTAL_PROCESSED_FRAMES = 100 * 10**6 # 100 million frames

with open('.env', mode='r', encoding='utf-8') as f:
    target_path = "EXPERIMENT/" + f.readline().replace('\n', '')
print("TARGET : {}".format(target_path.replace("EXPERIMENT/", "")))

json_open = open(target_path + "/"+ "param.json", "r")
json_load = json.load(json_open)
TOTAL_PROCESSED_FRAMES = json_load['total_step']
print("(TOTAL_PROCESSED_FRAMES) : {}".format(TOTAL_PROCESSED_FRAMES)) #add

class TrainingSaver:
    def __init__(self, shared_network, scene_networks, optimizer, config, SSL):
        self.config = config
        self.SSL = SSL
        #n_config = DEFAULT_CONFIG.copy()
        #n_config.update(config)
        #self.config.update(n_config)
        #self.config = config
        self.checkpoint_path = self.config['checkpoint_path']
        # self.saving_period = self.config['saving_period'] #origin
        self.saving_period = self.config['saving_period']//5 #add
        self.shared_network = shared_network
        self.scene_networks = scene_networks
        self.optimizer = optimizer 
        self.save_count = 0       

    def after_optimization(self,id):
        if id == 0:
            iteration = self.optimizer.get_global_step()
            if iteration >= self.save_count*self.saving_period:
                print('Saving training session')
                self.save()
                self.save_count = self.save_count + 1
        #iteration = self.optimizer.get_global_step()
        #if iteration % self.saving_period == 0:
        #    self.save()

    def print_config(self, offset: int = 0):
        for key, val in self.config.items():
            print((" " * offset) + f"{key}: {val}")
        pass

    def save(self):
        iteration = self.optimizer.get_global_step()
        # filename = self.checkpoint_path.replace('{checkpoint}', str(iteration)) # origin
        filename = self.checkpoint_path.replace('{checkpoint}', str(iteration*5)) #add
        model = dict()
        model['navigation'] = self.shared_network.state_dict()
        if self.SSL:
            for key, val in self.scene_networks.items():
                model[f'navigation/{key}'] = val.state_dict()
        else:
            model['navigation/scene'] = self.scene_networks.state_dict()
        model['optimizer'] = self.optimizer.state_dict()
        model['config'] = self.config
        
        with suppress(FileExistsError):
            os.makedirs(os.path.dirname(filename))
        torch.save(model, open(filename, 'wb'))

    def restore(self, state):
        if 'optimizer' in state and self.optimizer is not None:
            self.optimizer.load_state_dict(state['optimizer'])
        if 'config' in state:
            conf = self.config
            self.config = state['config']
            for k, v in conf.items():
                self.config[k] = v
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in state['navigation'].items():
            if "net." in key:
                new_state_dict = state['navigation']
                break
            new_state_dict['net.'+key] = value
        self.shared_network.load_state_dict(new_state_dict)

        self.scene_networks.load_state_dict(
            state[f'navigation/scene'])
        #if 'optimizer' in state and self.optimizer is not None: self.optimizer.load_state_dict(state['optimizer'])
        #if 'config' in state: 
        #    n_config = state['config'].copy()
        #    n_config.update(self.config)
        #    self.config.update(n_config)

        #self.shared_network.load_state_dict(state['navigation'])

        #tasks = self.config.get('tasks', TASK_LIST)
        #if self.SSL:
        #    for scene in tasks.keys():
        #        self.scene_networks[scene].load_state_dict(state[f'navigation/{scene}'])
        #else:
        #    self.scene_networks.load_state_dict(state[f'navigation/scene'])

class TrainingOptimizer:
    def __init__(self, grad_norm, optimizer, scheduler):
        self.optimizer : torch.optim.Optimizer = optimizer
        self.scheduler = scheduler
        self.grad_norm = grad_norm
        self.global_step = torch.tensor(0)
        self.lock = mp.Lock()

    def state_dict(self):
        state_dict = dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        state_dict["global_step"] = self.global_step
        # print(f"state_dict : {}".format(state_dict)) #add
        return state_dict

    def share_memory(self):
        self.global_step.share_memory_()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.global_step.copy_(state_dict['global_step'])
    
    def get_global_step(self):
        return self.global_step.item()

        
    def _ensure_shared_grads(self, local, shared, gpu = False):
        for param, shared_param in zip(filter(lambda p: p.requires_grad, local.parameters()),filter(lambda p: p.requires_grad, shared.parameters())):
            if shared_param.grad is not None and not gpu:
                return
            elif not gpu:
                shared_param._grad = param.grad
            else:
                shared_param._grad = param.grad.cpu()

    def optimize(self, loss, local, shared, gpu):

        # Fix the optimizer property after unpickling
        self.scheduler.optimizer = self.optimizer
        self.scheduler.step(self.global_step.item())

        # Increment step
        with self.lock:
            self.global_step.copy_(torch.tensor(self.global_step.item() + 1))
        local.zero_grad()           
        self.optimizer.zero_grad()

        # Calculate the new gradient with the respect to the local network
        loss.backward()

        # Clip gradient
        torch.nn.utils.clip_grad_norm_(list(local.parameters()), self.grad_norm)
        self._ensure_shared_grads(local,shared,gpu)
        self.optimizer.step()

class AnnealingLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        #self.max_t = max_t
        super(AnnealingLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        #num_warmup_steps = 200000
        #if self.last_epoch < num_warmup_steps:
        #    return [(float(self.last_epoch) / float(max(1.0, num_warmup_steps))) * base_lr for base_lr in self.base_lrs]
        return [base_lr * (1.0 - self.last_epoch / self.total_epochs)  
                for base_lr in self.base_lrs]                           ####note this is different Raph's work
        #return [base_lr * (1.0 - (self.last_epoch * self.max_t) / self.total_epochs)
        #        for base_lr in self.base_lrs]

class Training:
    if json_load["restore"]:
        def __init__(self, device, SSL, NGPU,NThreads, method, config):
            if torch.cuda.is_available() and config.get('cuda'):
                self.cuda = True
            else:
                self.cuda = False
            #self.cuda = config.get('cuda')
            self.SSL = config.get('SSL')
            self.Posi = config.get('Posi')
            self.Key = config.get('Key')
            self.NGPU = config.get('NGPU')
            self.NThreads = config.get('num_thread')
            self.method = config.get('method')
            self.tasks = config.get('task_list')
            self.action_size = config.get('action_size')
            #self.device = device    #json
            #self.SSL = SSL          #json   
            #self.NGPU = NGPU    #json
            #self.NThreads = NThreads #json
            #self.method = method #json
            #if self.method == 'word2vec_notarget':
            #    self.action_size = 5
            #else:
            #    self.action_size = 4
            self.log_path = config.get('log_path')
            self.config = config
            self.logger : logging.Logger = self._init_logger()
            self.learning_rate = config.get('learning_rate')
            self.rmsp_alpha = config.get('rmsp_alpha')
            self.rmsp_epsilon = config.get('rmsp_epsilon')
            self.grad_norm = config.get('grad_norm', 40.0)
            #self.tasks = config.get('tasks', TASK_LIST)
            self.checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
            #self.max_t = config.get('max_t', 5)
            self.max_t = config.get('max_t')
            #self.total_epochs = TOTAL_PROCESSED_FRAMES // self.max_t
            self.total_epochs = config.get('total_step') // self.max_t
            #self.total_epochs = TOTAL_PROCESSED_FRAMES
            self.initialize()
    else:
        def __init__(self, config): # from zero 
            if torch.cuda.is_available() and config.get('cuda'):
                self.cuda = True
            else:
                self.cuda = False
            #self.cuda = config.get('cuda')
            self.SSL = config.get('SSL')
            self.Posi = config.get('Posi')
            self.Key = config.get('Key')
            self.NGPU = config.get('NGPU')
            self.NThreads = config.get('num_thread')
            self.method = config.get('method')
            self.tasks = config.get('task_list')
            self.action_size = config.get('action_size')
            #self.device = device    #json
            #self.SSL = SSL          #json   
            #self.NGPU = NGPU    #json
            #self.NThreads = NThreads #json
            #self.method = method #json
            #if self.method == 'word2vec_notarget':
            #    self.action_size = 5
            #else:
            #    self.action_size = 4
            self.log_path = config.get('log_path')
            self.config = config
            self.logger : logging.Logger = self._init_logger()
            self.learning_rate = config.get('learning_rate')
            self.rmsp_alpha = config.get('rmsp_alpha')
            self.rmsp_epsilon = config.get('rmsp_epsilon')
            self.grad_norm = config.get('grad_norm', 40.0)
            #self.tasks = config.get('tasks', TASK_LIST)
            self.checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
            #self.max_t = config.get('max_t', 5)
            self.max_t = config.get('max_t')
            #self.total_epochs = TOTAL_PROCESSED_FRAMES // self.max_t
            self.total_epochs = config.get('total_step') // self.max_t
            #self.total_epochs = TOTAL_PROCESSED_FRAMES
            self.initialize()

    @staticmethod
    def load_checkpoint(device, SSL, NGPU, NThreads, method, config, fail = True):
        #device = torch.device('cpu')
        #if torch.cuda.is_available():
        #    device = True
        #else:
        #    device = False
        checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        max_t = config.get('max_t', 5)
        total_epochs = TOTAL_PROCESSED_FRAMES // max_t
        #total_epochs = TOTAL_PROCESSED_FRAMES
        files = os.listdir(os.path.dirname(checkpoint_path))
        base_name = os.path.basename(checkpoint_path)
        
        # Find latest checkpoint
        # TODO: improve speed
        restore_point = None
        if base_name.find('{checkpoint}') != -1:
            regex = re.escape(base_name).replace(re.escape('{checkpoint}'), '(\d+)')
            points = [(fname, int(match.group(1))) for (fname, match) in ((fname, re.match(regex, fname),) for fname in files) if not match is None]
            if len(points) == 0:
                if fail:
                    raise Exception('Restore point not found')
                else: return None
            
            (base_name, restore_point) = max(points, key = lambda x: x[1])

            
        print(f'Restoring from checkpoint {restore_point}')
        state = torch.load(open(os.path.join(os.path.dirname(checkpoint_path), base_name), 'rb'))
        # print("&&&&& (state) &&&&& : {}".format(state)) #test add
        
        if json_load["restore"]:
        # training = Training(device, SSL, NGPU, NThreads, method, state['config'] if 'config' in state else config) #origin
            training = Training(device, SSL, NGPU, NThreads, method, config) #add
        else:
            training = Training(device, state['config'] if 'config' in state else config) #origin
        training.saver.restore(state) 
        print('Configuration')
        training.saver.print_config(offset = 4)       
        return training

    def initialize(self):
        if self.SSL:
            # Shared network
            self.shared_network = SharedNetwork()
            self.scene_networks = { key:SceneSpecificNetwork(4) for key in TASK_LIST.keys() }

            # Share memory
            self.shared_network.share_memory()
            for net in self.scene_networks.values():
                net.share_memory()

            # Callect all parameters from all networks
            parameters = list(self.shared_network.parameters())
            for net in self.scene_networks.values():
                parameters.extend(net.parameters())

            # Create optimizer
            optimizer = SharedRMSprop(parameters, eps=self.rmsp_epsilon, alpha=self.rmsp_alpha, lr=self.learning_rate)
            optimizer.share_memory()

            # Create scheduler
            scheduler = AnnealingLRScheduler(optimizer, self.total_epochs)

            # Create optimizer wrapper
            optimizer_wrapper = TrainingOptimizer(self.grad_norm, optimizer, scheduler)
            self.optimizer = optimizer_wrapper
            optimizer_wrapper.share_memory()

            # Initialize saver
            self.saver = TrainingSaver(self.shared_network, self.scene_networks, self.optimizer, self.config, self.SSL)
        else:
            # Shared network
            #self.shared_network = SharedNetwork()
            self.shared_network = SharedNetwork(self.method)
            self.scene_networks = SceneSpecificNetwork(self.action_size, self.method) #Num. is num. of Action Space

            # Share memory
            self.shared_network.share_memory()
            self.scene_networks.share_memory()

            # Callect all parameters from all networks
            parameters = list(self.shared_network.parameters())
            parameters.extend(self.scene_networks.parameters())

            # Create optimizer
            optimizer = SharedRMSprop(parameters, eps=self.rmsp_epsilon, alpha=self.rmsp_alpha, lr=self.learning_rate)
            #optimizer = SharedAdam(parameters, eps=self.rmsp_epsilon, lr=self.learning_rate)
            optimizer.share_memory()

            # Create scheduler
            scheduler = AnnealingLRScheduler(optimizer, self.total_epochs)
            #scheduler = AnnealingLRScheduler(optimizer, self.total_epochs, self.max_t)

            # Create optimizer wrapper
            optimizer_wrapper = TrainingOptimizer(self.grad_norm, optimizer, scheduler)
            self.optimizer = optimizer_wrapper
            optimizer_wrapper.share_memory()

            # Initialize saver
            self.saver = TrainingSaver(self.shared_network, self.scene_networks, self.optimizer, self.config, self.SSL)

    
    def run(self):
        self.logger.info("Training started")

        # Prepare threads
        print(self.method)
        if self.method=='word2vec_notarget' or self.method=='Transformer_word2vec_notarget'or self.method=='Transformer_word2vec_notarget_withposi'or self.method=='Transformer_word2vec_notarget_word2vec'or self.method=='Transformer_word2vec_notarget_word2vec_posi' or self.method=="gcn_transformer" or self.method =="Transformer_word2vec_notarget_word2vec_concat" or self.method == 'Transformer_word2vec_notarget_word2vec_action' or self.method =='grid_memory' or self.method=='Transformer_word2vec_notarget_action'or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=='grid_memory_action':
            branches = []
            for scene in self.tasks.keys():
                it = 0
                for target in self.tasks.get(scene):
                    target['id'] = it
                    it = it + 1
                    branches.append((scene, target))
        else:
            if self.SSL:
                branches = [scene for scene in TASK_LIST.keys()]
            else:
                branches = [(scene, int(target)) for scene in self.tasks.keys() for target in self.tasks.get(scene)]
                #branches = [(scene, int(target)) for scene in TASK_LIST.keys() for target in TASK_LIST.get(scene)]
        print(branches)
        
        def _createThread(id, task,device):
            net = nn.Sequential(self.shared_network, self.scene_networks)
            net.share_memory()
            return TrainingThread(
                id = id,
                optimizer = self.optimizer,
                network = net,
                device = device,
                saver = self.saver,
                tasks = task,
                method = self.method,
                action_size = self.action_size,
                **self.config)

        def _createThreadwithque(id, task,device,summary_queue):
            net = nn.Sequential(self.shared_network, self.scene_networks)
            net.share_memory()
            return TrainingThread(
                id = id,
                optimizer = self.optimizer,
                network = net,
                device = device,
                saver = self.saver,
                tasks = task,
                #method = self.method,
                #action_size = self.action_size,
                summary_queue=summary_queue,
                **self.config)

        def _createThreadwithSSL(id, task, device, Ndevice):
            scene = task
            tasks = TASK_LIST.get(scene)
            if device:
                device_id = id % Ndevice
                device = torch.device("cuda:"+str(device_id))
            else:
                device = torch.device("cpu")
            net = nn.Sequential(self.shared_network, self.scene_networks[scene])
            net.share_memory()
            return TrainingThread(
                id = id,
                optimizer = self.optimizer,
                network = net,
                device = self.device,
                scene = scene,
                saver = self.saver,
                tasks = tasks,
                **self.config)

        if self.SSL:        # not increase threads more than number of scene
            if self.cuda:
                self.threads = [_createThreadwithSSL(i, scene, True, self.NGPU) for i, scene in enumerate(branches)]
            else:
                self.threads = [_createThreadwithSSL(i, scene, False, 1) for i, scene in enumerate(branches)]
            try:
                for thread in self.threads:
                    thread.start()

                for thread in self.threads:
                    thread.join()
                    
            except KeyboardInterrupt:
                # we will save the training
                print('Saving training session')
                self.saver.save()
        else:
            if self.method == 'word2vec_notarget' or self.method=='Baseline' or self.method=='Transformer_word2vec_notarget' or self.method == 'Transformer_Concat'or self.method=='Transformer_word2vec_notarget_withposi'or self.method=='Transformer_word2vec_notarget_word2vec'or self.method=='Transformer_word2vec_notarget_word2vec_posi'or self.method=="gcn_transformer" or self.method =="Transformer_word2vec_notarget_word2vec_concat"or self.method == 'Transformer_word2vec_notarget_word2vec_action'or self.method =='grid_memory'or self.method=='Transformer_word2vec_notarget_action'or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=='grid_memory_action':
                self.threads = []
                # Queues will be used to pass info to summary thread
                summary_queue = mp.Queue()

                # Create a summary thread to log
                actions = THORDiscreteEnvironment.acts[:self.action_size]
                self.summary = SummaryThread(
                    self.log_path, summary_queue, actions)
                del actions

                # self.threads = [_createThread(i, task) for i, task in enumerate(branches)]
                # print(f"Running for {self.total_epochs}") #origin
                print(f"Running for {self.total_epochs*5}") #add
                try:
                    # Start the logger thread
                    self.summary.start()

                    if self.cuda:
                        for i in range(self.NThreads):
                            j = i % self.NGPU
                            print(j)
                            device = torch.device("cuda:"+str(j))
                            print(device)
                            self.threads.append(_createThreadwithque(i, branches,device,summary_queue))
                            self.threads[-1].start()
                            time.sleep(2)
                    else:
                        for i in range(self.NThreads):
                            device = torch.device("cpu")
                            self.threads.append(_createThreadwithque(i, branches,device,summary_queue))
                            self.threads[-1].start()
                    # Wait for agent
                    for thread in self.threads:
                        thread.join()

                    # Wait for logger
                    self.summary.stop()
                    self.summary.join()

                    # Save last checkpoint
                    self.saver.save()
                except KeyboardInterrupt:
                    # we will save the training
                    print('Saving training session')
                    self.saver.save()

                    for thread in self.threads:
                        thread.stop()
                        thread.join()

                    self.summary.stop()
                    self.summary.join()
            else:
                self.threads = []
                try:
                    if self.cuda:
                        for i in range(self.NThreads):
                            j = i % self.NGPU
                            print(j)
                            device = torch.device("cuda:"+str(j))
                            print(device)
                            self.threads.append(_createThread(i, branches,device))
                            self.threads[-1].start()
                            time.sleep(2)
                    else:
                        for i in range(self.NThreads):
                            device = torch.device("cpu")
                            self.threads.append(_createThread(i, branches,device))
                            self.threads[-1].start()

                    for thread in self.threads:
                        thread.join()

                except KeyboardInterrupt:
                    # we will save the training
                    print('Saving training session')
                    self.saver.save()
        

    def _init_logger(self):
        logger = logging.getLogger('agent')
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        return logger

