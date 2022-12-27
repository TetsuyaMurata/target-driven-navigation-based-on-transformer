import torch.nn as nn
import torch.nn.functional as F
import torch
from agent.resnet import resnet50
import numpy as np
import math
import h5py
import json
from torch.nn.parameter import Parameter
import scipy.sparse as sp
import torchvision.models as models
from agent import transformer as TF

with open('.env', mode='r', encoding='utf-8') as f:
    target_path = "EXPERIMENT/" + f.readline().replace('\n', '')

json_open = open(target_path + "/"+ "param.json", "r")
json_load = json.load(json_open)

memory_size_read = json_load['memory']
print("(network) memory_size : {}".format(str(memory_size_read))) #test

class SharedNetwork(nn.Module):
    """ Bottom network, will extract feature for the policy network
    """
    def __init__(self, method):
        super(SharedNetwork, self).__init__()
        self.method = method
        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        mask_size=16####

        if self.method == 'Baseline': #zhu 
            self.net = Target_driven()
        elif self.method == 'Transformer_Sum': # zhu
            self.net = TransformerSum()
        elif self.method == 'Transformer_Concat':
            self.net = TransformerConcat()
        elif self.method == 'Transformer_Sum_without_current':
            self.net = TransformerSum_Without_Current()
        elif self.method =="word2vec_notarget": #raph
            self.net = word2vec_notarget(method, mask_size=mask_size)
        elif self.method =="Transformer_word2vec_notarget": #SMT
            self.net = Transformer_word2vec_notarget(method, mask_size=mask_size)
        elif self.method =="Transformer_word2vec_notarget_action":
            self.net = Transformer_word2vec_notarget_action(method, mask_size=mask_size)
        elif self.method =="Transformer_word2vec_notarget_withposi":
            self.net = Transformer_word2vec_notarget_withposi(method, mask_size=mask_size)
        elif self.method =="Transformer_word2vec_notarget_word2vec":
            self.net = Transformer_word2vec_notarget_word2vec(method, mask_size=mask_size)
        elif self.method =="Transformer_word2vec_notarget_word2vec_concat":
            self.net = Transformer_word2vec_notarget_word2vec_concat(method, mask_size=mask_size)
        elif self.method =="Transformer_word2vec_notarget_word2vec_posi":
            self.net = Transformer_word2vec_notarget_word2vec_posi(method, mask_size=mask_size)
        elif self.method =="Transformer_word2vec_notarget_word2vec_action":
            self.net = Transformer_word2vec_notarget_word2vec_action(method, mask_size=mask_size)
        elif self.method =="Transformer_word2vec_notarget_word2vec_action_posi":
            self.net = Transformer_word2vec_notarget_word2vec_action_posi(method, mask_size=mask_size)
        elif self.method =="gcn_transformer": 
            self.net = gcn_transformer(method, mask_size=mask_size)
        elif self.method =="grid_memory": #OMT
            self.net = Grid_memory(method, mask_size=mask_size)
        elif self.method =="grid_memory_no_observation":
            self.net = Grid_memory_no_observation(method, mask_size=mask_size)
        elif self.method =="grid_memory_action":
            self.net = Grid_memory_action(method, mask_size=mask_size)
        elif self.method =="scene_only": #add
            self.net = Scene_only(method, mask_size=mask_size)
        else:
            raise Exception("Please choose a method")

    #def save_gradient(self, grad):
    #    self.gradient = grad

    #def hook_backward(self, module, grad_input, grad_output):
    #    self.gradient_vanilla = grad_input[0]

    def save_gradient(self, grad):
        self.gradient = grad
    
    def forward(self, inp):
        return self.net(inp)


class Target_driven(nn.Module):
    """Target driven using visual input as target
    """ 
    def __init__(self):
        super(Target_driven, self).__init__()
        
        # Siemense layer
        self.fc_siemense= nn.Linear(8192, 512)
        #self.fc_siemense= nn.Linear(2048, 512)

        # Merge layer
        self.fc_merge = nn.Linear(1024, 512)
    
    def forward(self, inp):
        (x, y,) = inp
        
        x = x.view(-1)
        x = self.fc_siemense(x)  
        x = F.relu(x, True)

        y = y.view(-1)
        y = self.fc_siemense(y)
        y = F.relu(y, True)

        xy = torch.stack([x,y], 0).view(-1)
        xy = self.fc_merge(xy)
        xy = F.relu(xy, True)
        return xy


class TransformerConcat(nn.Module):
    def __init__(self):
        super(TransformerConcat, self).__init__()

        # Siemense layer
        self.fc_siemense= nn.Linear(2049, 512) #PosEnco
        self.e = math.e

        # Merge layer
        self.fc_merge = nn.Linear(1024, 512)

        #Transformer
        self.transformer_model = nn.Transformer(d_model=512,nhead=8, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=512)
        
        #position encoding
        self.pos_encoder = PositionalEncoding_Concat(1)#Concat
        self.pos_encoder2 = PositionalEncoding_Concat_Deco(1)#Concat

    def forward(self, inp):
        (x, y, memory, mask, device) = inp
        
        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        emmemory = self.pos_encoder(memory)
        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)
        em_memory=torch.cat([j,torch.zeros(128-i,512).to(device)]).view(128, 1, -1)
        
        x = x.view(1,-1)
        x = self.pos_encoder2(x)
        x = x.view(-1)
        x = self.fc_siemense(x)
        x = F.relu(x, True)
        y = y.view(1,-1)
        y = self.pos_encoder2(y)#????
        y = y.view(-1)
        y = self.fc_siemense(y)
        y = F.relu(y, True)

        xy = torch.stack([x,y], 0).view(-1)
        xy = self.fc_merge(xy)
        xy = F.relu(xy, True)
        xy = xy.view(1, 1, -1)
        xy = self.transformer_model(em_memory, xy, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        return (xy, memory, mask)

class TransformerSum(nn.Module):
    def __init__(self):
        super(TransformerSum, self).__init__()

        # Siemense layer
        self.fc_siemense= nn.Linear(2048, 512)
        self.e = math.e

        # Merge layer
        self.fc_merge = nn.Linear(1024, 512)

        #Transformer
        self.transformer_model = nn.Transformer(d_model=512,nhead=8, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=512)
        
        #position encoding
        self.pos_encoder = PositionalEncoding_Sum(512) #Sum

    def forward(self, inp):
        (x, y, memory, mask, device) = inp
        
        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        em_memory = torch.zeros(128,512).to(device)
        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(memory[0:i]),True).view(i,-1)
        em_memory=torch.cat([j,torch.zeros(128-i,512).to(device)])##could be one line
        em_memory = self.pos_encoder(em_memory)
        em_memory = em_memory.view(128, 1, -1)
        
        x = x.view(-1)
        x = self.fc_siemense(x)
        x = F.relu(x, True)
        y = y.view(-1)
        y = self.fc_siemense(y)
        y = F.relu(y, True)

        xy = torch.stack([x,y], 0).view(-1)
        xy = self.fc_merge(xy)
        xy = F.relu(xy, True)
        xy = xy.view(1, 1, -1)
        xy = self.transformer_model(em_memory, xy, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        return (xy, memory, mask)


class TransformerSum_Without_Current(nn.Module):
    def __init__(self):
        super(TransformerSum_Without_Current, self).__init__()

        # Siemense layer
        self.fc_siemense= nn.Linear(2048, 512)
        self.e = math.e

        # Merge layer
        self.fc_merge = nn.Linear(1024, 512)

        #Transformer
        self.transformer_model = nn.Transformer(d_model=512,nhead=8, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=512)
        
        #position encoding
        self.pos_encoder = PositionalEncoding_Sum(512) #Sum

    def forward(self, inp):
        (x, y, memory, mask, device) = inp
        
        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        em_memory = torch.zeros(128,512).to(device)
        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(memory[0:i]),True).view(i,-1)
        em_memory=torch.cat([j,torch.zeros(128-i,512).to(device)])##could be one line
        em_memory = self.pos_encoder(em_memory)
        em_memory = em_memory.view(128, 1, -1)
        
        y = y.view(-1)
        y = self.fc_siemense(y)
        y = F.relu(y, True)
        y = y.view(1,-1)
        y = self.pos_encoder(y)
        y = y.view(1, 1, -1)
        xy = self.transformer_model(em_memory, y, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        return (xy, memory, mask)

class SceneSpecificNetwork(nn.Module):
    """
    Input for this network is 512 tensor
    """
    def __init__(self, action_space_size, method):
        super(SceneSpecificNetwork, self).__init__()
        self.method = method
        # if self.method =="Transformer_word2vec_notarget_word2vec"or self.method =="Transformer_word2vec_notarget_word2vec_posi" or self.method =='gcn_transformer' or self.method =="Transformer_word2vec_notarget_word2vec_concat" or self.method =='Transformer_word2vec_notarget_word2vec_action' or self.method =='grid_memory' or self.method =='Transformer_word2vec_notarget_word2vec_action_posi' or self.method =='grid_memory_action': #origin
        if self.method =="Transformer_word2vec_notarget_word2vec"or self.method =="Transformer_word2vec_notarget_word2vec_posi" or self.method =='gcn_transformer' or self.method =="Transformer_word2vec_notarget_word2vec_concat" or self.method =='Transformer_word2vec_notarget_word2vec_action' or self.method =='grid_memory' or self.method=='grid_memory_no_observation' or self.method =='Transformer_word2vec_notarget_word2vec_action_posi' or self.method =='grid_memory_action' or self.method =='scene_only':

            self.fc1 = nn.Linear(300, 300)
            #self.fc1 = nn.Linear(912, 512)

            # Policy layer
            #self.fc2_policy = nn.Linear(512, action_space_size)
            self.fc2_policy = nn.Linear(300, action_space_size)

            # Value layer
            self.fc2_value = nn.Linear(300, 1)
            #self.fc2_value = nn.Linear(512, 1)
        else:
            self.fc1 = nn.Linear(512, 512)

            # Policy layer
            self.fc2_policy = nn.Linear(512, action_space_size)

            # Value layer
            self.fc2_value = nn.Linear(512, 1)

    def forward(self, inp):
        if self.method == "Baseline" or self.method =='word2vec_notarget':
            x = inp
            x = self.fc1(x)
            x = F.relu(x)
            x_policy = self.fc2_policy(x)
            #x_policy = F.softmax(x_policy)

            x_value = self.fc2_value(x)[0]
            return (x_policy, x_value, )
        elif self.method == "gcn_transformer":
            (x, memory, mask, gcn_memory, gcn_mask) = inp
            x = x.view(-1)
            x = self.fc1(x)
            x = F.relu(x)
            x_policy = self.fc2_policy(x)
            #x_policy = F.softmax(x_policy)

            x_value = self.fc2_value(x)[0]
            return (x_policy, x_value, memory, mask, gcn_memory, gcn_mask)
        elif self.method == "Transformer_word2vec_notarget_word2vec_action" or self.method == 'Transformer_word2vec_notarget_action'or self.method =='Transformer_word2vec_notarget_word2vec_action_posi':
            (x, memory, mask, act_memory, act_mask, encoder_atten_weights, decoder_atten_weights) = inp
            x = x.view(-1)
            x = self.fc1(x)
            x = F.relu(x)
            x_policy = self.fc2_policy(x)

            x_value = self.fc2_value(x)[0]
            return (x_policy, x_value, memory, mask, act_memory, act_mask, encoder_atten_weights, decoder_atten_weights)
        # elif self.method == "grid_memory": #origin
        elif self.method == "grid_memory" or self.method == "grid_memory_no_observation" or self.method == "scene_only":
            (x, memory, mask, grid_memory, grid_mask, encoder_atten_weights, decoder_atten_weights) = inp
            x = x.view(-1)
            x = self.fc1(x)
            x = F.relu(x)
            x_policy = self.fc2_policy(x)

            x_value = self.fc2_value(x)[0]
            return (x_policy, x_value, memory, mask, grid_memory, grid_mask, encoder_atten_weights, decoder_atten_weights)
        elif self.method == "grid_memory_action":
            (x, memory, mask, grid_memory, grid_mask, act_memory, act_mask, encoder_atten_weights, decoder_atten_weights) = inp
            x = x.view(-1)
            x = self.fc1(x)
            x = F.relu(x)
            x_policy = self.fc2_policy(x)

            x_value = self.fc2_value(x)[0]
            return (x_policy, x_value, memory, mask, grid_memory, grid_mask, act_memory, act_mask, encoder_atten_weights, decoder_atten_weights)
        else:
            (x, memory, mask, encoder_atten_weights, decoder_atten_weights) = inp
            x = x.view(-1)
            x = self.fc1(x)
            x = F.relu(x)
            x_policy = self.fc2_policy(x)

            x_value = self.fc2_value(x)[0]
            return (x_policy, x_value, memory, mask, encoder_atten_weights, decoder_atten_weights)

class word2vec_notarget(nn.Module):
    """Our method network without target word embedding
    """

    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=16):
        super(word2vec_notarget, self).__init__()

        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None

        # Observation layer
        self.fc_observation = nn.Linear(8192, 512)

        # Convolution for similarity grid
        pooling_kernel = 2
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1.register_backward_hook(self.hook_backward)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        conv1_output = (mask_size - 3 + 1)//pooling_kernel
        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        self.flat_input = 16 * conv2_output * conv2_output

        # Merge layer
        self.fc_merge = nn.Linear(
            512+self.flat_input, 512)

    def forward(self, inp):
        # x is the observation
        # z is the object location mask
        (x, z) = inp
        x = x.view(-1)
        x = self.fc_observation(x)
        x = F.relu(x, True)
        z = torch.autograd.Variable(z, requires_grad=True)
        z = self.conv1(z)
        z.register_hook(self.save_gradient)
        self.conv_output = z
        z = self.pool(F.relu(z))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.view(-1)
        self.output_context = z

        # xy = torch.stack([x, y], 0).view(-1)
        xyz = torch.cat([x, z])
        xyz = self.fc_merge(xyz)
        xyz = F.relu(xyz, True)
        return xyz

#class Transformer_word2vec_notarget(nn.Module):
#    """Our method network without target word embedding
#    """
#
#    def save_gradient(self, grad):
#        self.gradient = grad
#
#    def hook_backward(self, module, grad_input, grad_output):
#        self.gradient_vanilla = grad_input[0]
#
#    def __init__(self, method, mask_size=16):
#        super(Transformer_word2vec_notarget, self).__init__()
#
#        # Siemense layer
#        #self.fc_siemense= nn.Linear(2049, 512) #PosEnco
#        self.fc_siemense= nn.Linear(2048, 512) #PosEnco
#        self.e = math.e
#
#        # Merge layer
#        #self.fc_merge = nn.Linear(1024, 512)
#
#        #Transformer
#        self.transformer_model = nn.Transformer(d_model=512,nhead=8, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=512)
#        
#        #position encoding
#        #self.pos_encoder = PositionalEncoding_Concat(1)#Concat
#        #self.pos_encoder2 = PositionalEncoding_Concat_Deco(1)#Concat
#
#        self.gradient = None
#        self.gradient_vanilla = None
#        self.conv_output = None
#        self.output_context = None
#
#        # Observation layer
#        #self.fc_observation = nn.Linear(8192, 512)
#        #self.fc_observation = nn.Linear(2049, 512)
#
#        # Convolution for similarity grid
#        pooling_kernel = 2
#        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
#        self.conv1.register_backward_hook(self.hook_backward)
#        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
#        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)
#
#        conv1_output = (mask_size - 3 + 1)//pooling_kernel
#        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
#        self.flat_input = 16 * conv2_output * conv2_output
#
#        # Merge layer
#        self.fc_merge = nn.Linear(
#            512+self.flat_input, 512)
#
#    def forward(self, inp):
#        (x, z, memory, mask, device) = inp
#        
#        x = x.view(-1)
#        memory = memory.to(device)
#        memory, mask = update_memory(x, memory, mask,device)
#        mask = mask.view(1,-1)
#        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
#        emmemory = memory
#        
#        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
#        j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)
#        em_memory=torch.cat([j,torch.zeros(memory_size_read-i,512).to(device)]).view(memory_size_read, 1, -1)
#        
#        x = x.view(-1)
#        x = self.fc_siemense(x)
#        x = F.relu(x, True)
#        x = x.view(1, 1, -1)
#
#        z = torch.autograd.Variable(z, requires_grad=True)
#        z = self.conv1(z)
#        z.register_hook(self.save_gradient)
#        self.conv_output = z
#        z = self.pool(F.relu(z))
#        z = self.pool(F.relu(self.conv2(z)))
#        z = z.view(-1)
#        self.output_context = z
#
#        #xyz = torch.cat([x, z])
#        #xyz = self.fc_merge(xyz)
#        #xyz = F.relu(xyz, True)
#        #xyz = xyz.view(1, 1, -1)
#        
#        #xy = self.transformer_model(em_memory, xyz, src_key_padding_mask = amask, memory_key_padding_mask = amask)
#        xy = self.transformer_model(em_memory, x, src_key_padding_mask = amask, memory_key_padding_mask = amask)
#
#        xy = xy.view(-1)
#        xy = torch.cat([xy, z])
#        xy = self.fc_merge(xy)
#        xy = F.relu(xy, True)
#        xy = xy.view(1, 1, -1)
#        return (xy, memory, mask)



class Transformer_word2vec_notarget(nn.Module):
    """Our method network without target word embedding
    """

    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=16):
        super(Transformer_word2vec_notarget, self).__init__()

        # Siemense layer
        self.fc_siemense= nn.Linear(2049, 512) #PosConcat
        #self.fc_siemense= nn.Linear(2048, 512) #PosSum
        self.e = math.e

        # Merge layer
        #self.fc_merge = nn.Linear(1024, 512)

        #Transformer
        #self.transformer_model = nn.Transformer(d_model=512,nhead=8, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=512)
        self.transformer_model = TF.Transformer(d_model=512,nhead=8, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=512)
        
        #position encoding
        self.pos_encoder = PositionalEncoding_Concat(1)#Concat
        self.pos_encoder2 = PositionalEncoding_Concat_Deco(1)#Concat
        #self.pos_encoder = PositionalEncoding_Sum(512) #Sum

        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None

        # Observation layer
        #self.fc_observation = nn.Linear(8192, 512)
        #self.fc_observation = nn.Linear(2049, 512)

        # Convolution for similarity grid
        pooling_kernel = 2
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1.register_backward_hook(self.hook_backward)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        conv1_output = (mask_size - 3 + 1)//pooling_kernel
        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        self.flat_input = 16 * conv2_output * conv2_output

        # Merge layer
        self.fc_merge = nn.Linear(
            512+self.flat_input, 512)

    def forward(self, inp):
        (x, z, memory, mask, device) = inp
        
        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        emmemory = memory
        ##SUM###
        #i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        #j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)
        #em_memory=torch.cat([j,torch.zeros(4-i,512).to(device)])
        #em_memory = self.pos_encoder(em_memory)
        #em_memory = em_memory.view(4, 1, -1)
        
        #x = x.view(-1)
        #x = self.fc_siemense(x)
        #x = F.relu(x, True)
        #x = x.view(1, 1, -1)
        
        ###Concat###
        em_memory = self.pos_encoder(emmemory)
        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(em_memory[0:i]),True).view(i,-1)
        em_memory=torch.cat([j,torch.zeros(1-i,512).to(device)])
        em_memory = em_memory.view(1, 1, -1)
        
        x = x.view(1,-1)
        x = self.pos_encoder2(x)
        x = x.view(-1)
        x = self.fc_siemense(x)
        x = F.relu(x, True)
        x = x.view(1, 1, -1)
        

        z = torch.autograd.Variable(z, requires_grad=True)
        z = self.conv1(z)
        z.register_hook(self.save_gradient)
        self.conv_output = z
        z = self.pool(F.relu(z))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.view(-1)
        self.output_context = z

        #xyz = torch.cat([x, z])
        #xyz = self.fc_merge(xyz)
        #xyz = F.relu(xyz, True)
        #xyz = xyz.view(1, 1, -1)
        
        #xy = self.transformer_model(em_memory, xyz, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        #xy = self.transformer_model(em_memory, x, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        print(em_memory.size())
        print(amask.sixe())

        xy, encoder_atten_weights, decoder_atten_weights = self.transformer_model(em_memory, x, src_key_padding_mask = amask, memory_key_padding_mask = amask)

        xy = xy.view(-1)
        xy = torch.cat([xy, z])
        xy = self.fc_merge(xy)
        xy = F.relu(xy, True)
        xy = xy.view(1, 1, -1)
        return (xy, memory, mask, encoder_atten_weights, decoder_atten_weights)


class Transformer_word2vec_notarget_action(nn.Module):
    """Our method network without target word embedding
    """
    def save_gradient(self, grad):
        self.gradient = grad
    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]
    def __init__(self, method, mask_size=16):
        super(Transformer_word2vec_notarget_action, self).__init__()

        # Siemense layer
        self.fc_siemense= nn.Linear(2048, 512) #PosSum
        #self.fc_siemense2= nn.Linear(2048, 512) #PosSum
        self.e = math.e
        #self.fc_action= nn.Linear(8, 16) #PosEnco
        self.fc_action= nn.Linear(128, 256) #PosEnco

        # Merge layer
        #self.fc_merge = nn.Linear(528, 512)
        self.fc_merge = nn.Linear(768, 512)

        #Transformer
        self.transformer_model = TF.Transformer(d_model=512,nhead=8, num_encoder_layers=3,num_decoder_layers=3,dim_feedforward=512)
        
        #position encoding
        self.pos_encoder = PositionalEncoding_Sum(512) #Sum

        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None

        # Convolution for similarity grid
        pooling_kernel = 2
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1.register_backward_hook(self.hook_backward)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        conv1_output = (mask_size - 3 + 1)//pooling_kernel
        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        self.flat_input = 16 * conv2_output * conv2_output

        # Merge layer
        self.fc_merge2 = nn.Linear(
            512+self.flat_input, 512)
        #Dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, inp):
        (x, z, memory, mask, device, last_act, act_memory, act_mask) = inp
        
        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        emmemory = memory

        last_act.view(-1)
        act_memory = act_memory.to(device)
        #act_memory, act_mask = update_memory(last_act, act_memory, act_mask,device)
        act_memory, act_mask = update_memory_action(last_act, act_memory, act_mask,device)
        act_mask = act_mask.view(1,-1)
        act_amask = torch.tensor(act_mask.clone().detach(), dtype=bool).to(device)
        act_emmemory = act_memory
        ##SUM###
        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        #j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)
        j = self.dropout(F.relu(self.fc_siemense(emmemory[0:i]),True)).view(i,-1)
        
        
        k = F.relu(self.fc_action(act_emmemory[0:i]),True).view(i,-1)
        
        em_memory = torch.cat([j, k], dim = 1)
        em_memory = self.fc_merge(em_memory)
        em_memory = F.relu(em_memory, True)
        em_memory = torch.cat([em_memory,torch.zeros(16-i,512).to(device)])
        em_memory = self.pos_encoder(em_memory)
        em_memory = em_memory.view(16, 1, -1)

        
        x = x.view(-1)
        x = self.fc_siemense(x)
        #x = self.fc_siemense2(x)
        x = F.relu(x, True)
        x = self.dropout(x)
        x = x.view(1, 1, -1)
        
        z = torch.autograd.Variable(z, requires_grad=True)
        z = self.conv1(z)
        z.register_hook(self.save_gradient)
        self.conv_output = z
        z = self.pool(F.relu(z))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.view(-1)
        self.output_context = z

        xy, encoder_atten_weights, decoder_atten_weights = self.transformer_model(em_memory, x, src_key_padding_mask = amask, memory_key_padding_mask = amask)

        xy = xy.view(-1)
        xy = torch.cat([xy, z])
        xy = self.fc_merge2(xy)
        xy = F.relu(xy, True)
        xy = xy.view(1, 1, -1)
        return (xy, memory, mask, act_memory, act_mask, encoder_atten_weights, decoder_atten_weights)

class Transformer_word2vec_notarget_withposi(nn.Module):
    """Our method network without target word embedding
    """

    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=16):
        super(Transformer_word2vec_notarget_withposi, self).__init__()

        # Siemense layer
        #self.fc_siemense= nn.Linear(2049, 512) #PosEnco
        self.fc_siemense= nn.Linear(2048, 512) #PosEnco
        self.e = math.e

        # Merge layer
        #self.fc_merge = nn.Linear(1024, 512)

        #Transformer
        self.transformer_model = nn.Transformer(d_model=512,nhead=8, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=512)
        
        #position encoding
        #self.pos_encoder = PositionalEncoding_Concat(1)#Concat
        #self.pos_encoder2 = PositionalEncoding_Concat_Deco(1)#Concat
        self.pos_encoder = PositionalEncoding_Sum(512) #Sum
        self.pos_encoder_posi = PositionalEncoding_posi(512) #Sum


        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None

        # Observation layer
        #self.fc_observation = nn.Linear(8192, 512)
        #self.fc_observation = nn.Linear(2049, 512)

        # Convolution for similarity grid
        pooling_kernel = 2
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1.register_backward_hook(self.hook_backward)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        conv1_output = (mask_size - 3 + 1)//pooling_kernel
        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        self.flat_input = 16 * conv2_output * conv2_output

        # Merge layer
        self.fc_merge = nn.Linear(
            512+self.flat_input, 512)

    def forward(self, inp):
        (x, z, memory, mask, device,positions) = inp
        
        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        emmemory = memory
        
        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)
        em_memory=torch.cat([j,torch.zeros(memory_size_read-i,512).to(device)])
        em_memory = self.pos_encoder(em_memory)
        em_memory = self.pos_encoder_posi(em_memory,positions)
        em_memory = em_memory.view(memory_size_read, 1, -1)
        
        x = x.view(-1)
        x = self.fc_siemense(x)
        x = F.relu(x, True)
        x = x.view(1,-1)
        x = self.pos_encoder(x)
        x = x.view(1, 1, -1)

        z = torch.autograd.Variable(z, requires_grad=True)
        z = self.conv1(z)
        z.register_hook(self.save_gradient)
        self.conv_output = z
        z = self.pool(F.relu(z))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.view(-1)
        self.output_context = z

        #xyz = torch.cat([x, z])
        #xyz = self.fc_merge(xyz)
        #xyz = F.relu(xyz, True)
        #xyz = xyz.view(1, 1, -1)
        
        #xy = self.transformer_model(em_memory, xyz, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        xy = self.transformer_model(em_memory, x, src_key_padding_mask = amask, memory_key_padding_mask = amask)

        xy = xy.view(-1)
        xy = torch.cat([xy, z])
        xy = self.fc_merge(xy)
        xy = F.relu(xy, True)
        xy = xy.view(1, 1, -1)
        return (xy, memory, mask)



class Transformer_word2vec_notarget_word2vec(nn.Module):
    """Our method network without target word embedding
    """

    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=16):
        super(Transformer_word2vec_notarget_word2vec, self).__init__()

        # Siemense layer
        self.fc_siemense= nn.Linear(2048, 300) #PosEnco
        self.e = math.e

        # Merge layer
        #self.fc_merge = nn.Linear(1024, 512)

        #Transformer
        #self.transformer_model = nn.Transformer(d_model=300,nhead=5, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=300)
        self.transformer_model = TF.Transformer(d_model=300,nhead=5, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=300)
        
        #position encoding
        self.pos_encoder = PositionalEncoding_Sum(300) #Sum


        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None

        # Observation layer
        #self.fc_observation = nn.Linear(8192, 512)
        #self.fc_observation = nn.Linear(2049, 512)

        # Convolution for similarity grid
        pooling_kernel = 2
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1.register_backward_hook(self.hook_backward)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        conv1_output = (mask_size - 3 + 1)//pooling_kernel
        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        self.flat_input = 16 * conv2_output * conv2_output

        # Merge layer
        self.fc_merge = nn.Linear(
            300+self.flat_input, 300)

    def forward(self, inp):
        (x, z, memory, mask, device,word2vec) = inp
        
        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        emmemory = memory
        
        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)
        em_memory=torch.cat([j,torch.zeros(4-i,300).to(device)])
        em_memory = self.pos_encoder(em_memory)
        em_memory = em_memory.view(4, 1, -1)
        
        word2vec = torch.from_numpy(word2vec).to(device)
        word2vec=word2vec.view(1,1,-1)

        z = torch.autograd.Variable(z, requires_grad=True)
        z = self.conv1(z)
        z.register_hook(self.save_gradient)
        self.conv_output = z
        z = self.pool(F.relu(z))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.view(-1)
        self.output_context = z

        xy, encoder_atten_weights, decoder_atten_weights = self.transformer_model(em_memory, word2vec, src_key_padding_mask = amask, memory_key_padding_mask = amask)

        xy = xy.view(-1)
        xy = torch.cat([xy, z])
        xy = self.fc_merge(xy)
        xy = F.relu(xy, True)
        xy = xy.view(1, 1, -1)
        return (xy, memory, mask, encoder_atten_weights, decoder_atten_weights)


class Transformer_word2vec_notarget_word2vec_action(nn.Module):
    """Our method network without target word embedding
    """

    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=16):
        super(Transformer_word2vec_notarget_word2vec_action, self).__init__()

        # Siemense layer
        self.fc_siemense= nn.Linear(2048, 300) #PosEnco
        self.fc_action= nn.Linear(8, 16) #PosEnco
        self.e = math.e

        # Merge layer
        self.fc_merge = nn.Linear(316, 300)

        #Transformer
        self.transformer_model = TF.Transformer(d_model=300,nhead=5, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=300)
        
        #position encoding
        self.pos_encoder = PositionalEncoding_Sum(300) #Sum


        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None

        # Convolution for similarity grid
        pooling_kernel = 2
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1.register_backward_hook(self.hook_backward)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        conv1_output = (mask_size - 3 + 1)//pooling_kernel
        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        self.flat_input = 16 * conv2_output * conv2_output

        # Merge layer
        self.fc_merge2 = nn.Linear(
            300+self.flat_input, 300)

    def forward(self, inp):
        (x, z, memory, mask, device, word2vec, last_act, act_memory, act_mask) = inp
        
        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        emmemory = memory

        last_act.view(-1)
        act_memory = act_memory.to(device)
        act_memory, act_mask = update_memory(last_act, act_memory, act_mask,device)
        act_mask = act_mask.view(1,-1)
        act_amask = torch.tensor(act_mask.clone().detach(), dtype=bool).to(device)
        act_emmemory = act_memory
        
        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)
        em_memory=torch.cat([j,torch.zeros(memory_size_read-i,300).to(device)])
        
        k = F.relu(self.fc_action(act_emmemory[0:i]),True).view(i,-1)
        act_em_memory=torch.cat([k,torch.zeros(memory_size_read-i,16).to(device)])
        
        em_memory = torch.cat([em_memory, act_em_memory], dim = 1)
        em_memory = self.fc_merge(em_memory)
        em_memory = F.relu(em_memory, True)
        em_memory = self.pos_encoder(em_memory)
        em_memory = em_memory.view(memory_size_read, 1, -1)

        word2vec = torch.from_numpy(word2vec).to(device)
        word2vec=word2vec.view(1,1,-1)

        z = torch.autograd.Variable(z, requires_grad=True)
        z = self.conv1(z)
        z.register_hook(self.save_gradient)
        self.conv_output = z
        z = self.pool(F.relu(z))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.view(-1)
        self.output_context = z

        #xy = self.transformer_model(em_memory, xyz, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        #xy = self.transformer_model(em_memory, word2vec, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        xy, encoder_atten_weights, decoder_atten_weights = self.transformer_model(em_memory, word2vec, src_key_padding_mask = amask, memory_key_padding_mask = amask)

        xy = xy.view(-1)
        xy = torch.cat([xy, z])
        xy = self.fc_merge2(xy)
        xy = F.relu(xy, True)
        xy = xy.view(1, 1, -1)
        return (xy, memory, mask, act_memory, act_mask, encoder_atten_weights, decoder_atten_weights)


class Transformer_word2vec_notarget_word2vec_action_posi(nn.Module):
    """Our method network without target word embedding
    """

    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=16):
        super(Transformer_word2vec_notarget_word2vec_action_posi, self).__init__()

        # Siemense layer
        self.fc_siemense= nn.Linear(2048, 300) #PosEnco
        self.fc_action= nn.Linear(8, 16) #PosEnco
        self.fc_position= nn.Linear(4, 16) #PosEncoug18_11-59-15
        self.e = math.e

        # Merge layer
        self.fc_merge = nn.Linear(332, 300)

        #Transformer
        self.transformer_model = TF.Transformer(d_model=300,nhead=5, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=300)
        
        #position encoding
        self.pos_encoder = PositionalEncoding_Sum(300) #Sum

        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None

        # Convolution for similarity grid
        pooling_kernel = 2
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1.register_backward_hook(self.hook_backward)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        conv1_output = (mask_size - 3 + 1)//pooling_kernel
        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        self.flat_input = 16 * conv2_output * conv2_output

        # Merge layer
        self.fc_merge2 = nn.Linear(
            300+self.flat_input, 300)

    def forward(self, inp):
        (x, z, memory, mask, device, word2vec, last_act, act_memory, act_mask, posi) = inp
        
        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        emmemory = memory

        last_act.view(-1)
        act_memory = act_memory.to(device)
        act_memory, act_mask = update_memory(last_act, act_memory, act_mask,device)
        act_mask = act_mask.view(1,-1)
        act_amask = torch.tensor(act_mask.clone().detach(), dtype=bool).to(device)
        act_emmemory = act_memory
        
        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)
        em_memory=torch.cat([j,torch.zeros(memory_size_read-i,300).to(device)])
        
        k = F.relu(self.fc_action(act_emmemory[0:i]),True).view(i,-1)
        act_em_memory=torch.cat([k,torch.zeros(memory_size_read-i,16).to(device)])

        posi_emmemory = torch.tensor(posi).float().to(device)

        l = F.relu(self.fc_position(posi_emmemory[0:i]),True).view(i,-1)
        posi_em_memory=torch.cat([l,torch.zeros(memory_size_read-i,16).to(device)])
        
        em_memory = torch.cat([em_memory, act_em_memory], dim = 1)
        em_memory = torch.cat([em_memory, posi_em_memory], dim = 1)
        em_memory = self.fc_merge(em_memory)
        em_memory = F.relu(em_memory, True)
        em_memory = self.pos_encoder(em_memory)
        em_memory = em_memory.view(memory_size_read, 1, -1)

        word2vec = torch.from_numpy(word2vec).to(device)
        word2vec=word2vec.view(1,1,-1)

        z = torch.autograd.Variable(z, requires_grad=True)
        z = self.conv1(z)
        z.register_hook(self.save_gradient)
        self.conv_output = z
        z = self.pool(F.relu(z))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.view(-1)
        self.output_context = z

        xy, encoder_atten_weights, decoder_atten_weights = self.transformer_model(em_memory, word2vec, src_key_padding_mask = amask, memory_key_padding_mask = amask)

        xy = xy.view(-1)
        xy = torch.cat([xy, z])
        xy = self.fc_merge2(xy)
        xy = F.relu(xy, True)
        xy = xy.view(1, 1, -1)
        return (xy, memory, mask, act_memory, act_mask, encoder_atten_weights, decoder_atten_weights)

class Transformer_word2vec_notarget_word2vec_concat(nn.Module):
    """Our method network without target word embedding
    """

    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=16):
        super(Transformer_word2vec_notarget_word2vec_concat, self).__init__()

        # Siemense layer
        self.fc_siemense= nn.Linear(2049, 300) #PosEnco
        self.e = math.e

        # Merge layer
        #self.fc_merge = nn.Linear(1024, 512)

        #Transformer
        #self.transformer_model = nn.Transformer(d_model=300,nhead=5, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=300)
        self.transformer_model = TF.Transformer(d_model=300,nhead=5, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=300)
        
        #position encoding
        self.pos_encoder = PositionalEncoding_Concat(1) #Concat


        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None

        # Observation layer
        #self.fc_observation = nn.Linear(8192, 512)
        #self.fc_observation = nn.Linear(2049, 512)

        # Convolution for similarity grid
        pooling_kernel = 2
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1.register_backward_hook(self.hook_backward)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        conv1_output = (mask_size - 3 + 1)//pooling_kernel
        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        self.flat_input = 16 * conv2_output * conv2_output

        # Merge layer
        self.fc_merge = nn.Linear(
            300+self.flat_input, 300)

    def forward(self, inp):
        (x, z, memory, mask, device,word2vec) = inp
        
        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        emmemory = memory
        
        emmemory = self.pos_encoder(emmemory)
        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)
        em_memory=torch.cat([j,torch.zeros(memory_size_read-i,300).to(device)])
        em_memory = em_memory.view(memory_size_read, 1, -1)
        
        word2vec = torch.from_numpy(word2vec).to(device)
        word2vec=word2vec.view(1,1,-1)

        z = torch.autograd.Variable(z, requires_grad=True)
        z = self.conv1(z)
        z.register_hook(self.save_gradient)
        self.conv_output = z
        z = self.pool(F.relu(z))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.view(-1)
        self.output_context = z

        #xyz = torch.cat([x, z])
        #xyz = self.fc_merge(xyz)
        #xyz = F.relu(xyz, True)
        #xyz = xyz.view(1, 1, -1)
        
        #xy = self.transformer_model(em_memory, xyz, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        #xy = self.transformer_model(em_memory, word2vec, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        xy, encoder_atten_weights, decoder_atten_weights = self.transformer_model(em_memory, word2vec, src_key_padding_mask = amask, memory_key_padding_mask = amask)

        xy = xy.view(-1)
        xy = torch.cat([xy, z])
        xy = self.fc_merge(xy)
        xy = F.relu(xy, True)
        xy = xy.view(1, 1, -1)
        return (xy, memory, mask, encoder_atten_weights, decoder_atten_weights)

class Transformer_word2vec_notarget_word2vec_posi(nn.Module):
    """Our method network without target word embedding
    """

    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=16):
        super(Transformer_word2vec_notarget_word2vec_posi, self).__init__()

        # Siemense layer
        self.fc_siemense= nn.Linear(2048, 300) #PosEnco
        self.e = math.e

        #Transformer
        #self.transformer_model = nn.Transformer(d_model=300,nhead=5, num_encoder_layers=3,num_decoder_layers=3,dim_feedforward=300)
        self.transformer_model = TF.Transformer(d_model=300,nhead=5, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=300)
        
        #position encoding
        self.pos_encoder = PositionalEncoding_Sum(300) #Sum
        self.pos_encoder_posi = PositionalEncoding_posi(300) #Sum


        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None

        # Convolution for similarity grid
        pooling_kernel = 2
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1.register_backward_hook(self.hook_backward)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        conv1_output = (mask_size - 3 + 1)//pooling_kernel
        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        self.flat_input = 16 * conv2_output * conv2_output

        # Merge layer
        self.fc_merge = nn.Linear(
            300+self.flat_input, 300)

    def forward(self, inp):
        (x, z, memory, mask, device,positions, word2vec) = inp
        
        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        emmemory = memory
        
        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)
        em_memory=torch.cat([j,torch.zeros(memory_size_read-i,300).to(device)])
        em_memory = self.pos_encoder(em_memory)
        em_memory = self.pos_encoder_posi(em_memory,positions)
        em_memory = em_memory.view(memory_size_read, 1, -1)
        
        word2vec = torch.from_numpy(word2vec).to(device)
        word2vec=word2vec.view(1,1,-1)

        z = torch.autograd.Variable(z, requires_grad=True)
        z = self.conv1(z)
        z.register_hook(self.save_gradient)
        self.conv_output = z
        z = self.pool(F.relu(z))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.view(-1)
        self.output_context = z

        #xy = self.transformer_model(em_memory, word2vec, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        xy, encoder_atten_weights, decoder_atten_weights = self.transformer_model(em_memory, word2vec, src_key_padding_mask = amask, memory_key_padding_mask = amask)

        xy = xy.view(-1)
        xy = torch.cat([xy, z])
        xy = self.fc_merge(xy)
        xy = F.relu(xy, True)
        xy = xy.view(1, 1, -1)
        return (xy, memory, mask, encoder_atten_weights, decoder_atten_weights)





class ActorCriticLoss(nn.Module):
    def __init__(self, entropy_beta):
        self.entropy_beta = entropy_beta
        pass

    def forward(self, policy, value, action_taken, temporary_difference, r):
        # Calculate policy entropy
        log_softmax_policy = torch.nn.functional.log_softmax(policy, dim=1)
        softmax_policy = torch.nn.functional.softmax(policy, dim=1)
        policy_entropy = softmax_policy * log_softmax_policy
        policy_entropy = -torch.sum(policy_entropy, 1)

        # Policy loss
        nllLoss = F.nll_loss(log_softmax_policy, action_taken, reduce=False)
        policy_loss = nllLoss * temporary_difference - policy_entropy * self.entropy_beta
        policy_loss = policy_loss.sum(0)

        # Value loss
        # learning rate for critic is half of actor's
        # Equivalent to 0.5 * l2 loss
        value_loss = (0.5 * 0.5) * F.mse_loss(value, r, size_average=False)
        return value_loss + policy_loss

def update_memory(observation, memory, mask, device):
    """
    Update the memory and mask based on latest observation
    """
    assert (
        observation.shape[0] == memory.shape[1]
    ), f"Embedding sizes don't match, {observation.shape[0]} vs {memory.shape[1]}"
    assert mask.shape[0] == memory.shape[0], f"Memory sizes don't match"
    #assert len(reset.shape.as_list()) == 0, f"Reset must be scalar"

    #reset = torch.cat(reset)

    # Reset memory if requested
    #new_memory = memory * (1 - reset)
    new_memory = memory 
    #print(observation)
    #print(new_memory)
    # Shift memory forward and add new observation
    new_memory = torch.cat(
        [observation.unsqueeze(0), new_memory[:-1,:]], axis=0
        )

    # Update mask
    #new_mask = mask * (1 - reset)
    new_mask = mask 
    #new_mask = torch.cat([torch.ones((1)), new_mask[:-1]], axis=0)
    new_mask = torch.cat([torch.zeros((1)), new_mask[:-1]], axis=0)
    #new_mask = torch.cat([torch.zeros((1)).to(device), new_mask[:-1]], axis=0)
    return new_memory, new_mask

def update_memory_action(observation, memory, mask, device):
    """
    Update the memory and mask based on latest observation
    """
    assert (
        observation.shape[0] == memory.shape[1]
    ), f"Embedding sizes don't match, {observation.shape[0]} vs {memory.shape[1]}"
    assert mask.shape[0] == memory.shape[0], f"Memory sizes don't match"
    #assert len(reset.shape.as_list()) == 0, f"Reset must be scalar"

    #reset = torch.cat(reset)

    # Reset memory if requested
    #new_memory = memory * (1 - reset)
    new_memory = memory 
    #print(observation)
    #print(new_memory)
    # Shift memory forward and add new observation
    new_memory = torch.cat(
        [observation.unsqueeze(0), new_memory[1:,:]], axis=0
        )

    new_memory = torch.cat(
        [torch.zeros((1, 128)).to(device), new_memory[:-1,:]], axis=0
        )
    # Update mask
    #new_mask = mask * (1 - reset)
    new_mask = mask 
    #new_mask = torch.cat([torch.ones((1)), new_mask[:-1]], axis=0)
    new_mask = torch.cat([torch.zeros((1)), new_mask[:-1]], axis=0)
    #new_mask = torch.cat([torch.zeros((1)).to(device), new_mask[:-1]], axis=0)
    return new_memory, new_mask


def update_gcn_memory(observation, memory, mask, device):
    """
    Update the memory and mask based on latest observation
    """
    assert (
        observation.shape[0] == memory.shape[1]
    ), f"Embedding sizes don't match, {observation.shape[0]} vs {memory.shape[1]}"
    assert mask.shape[0] == memory.shape[0], f"Memory sizes don't match"
    #assert len(reset.shape.as_list()) == 0, f"Reset must be scalar"

    new_memory = memory 
    # Shift memory forward and add new observation
    new_memory = torch.cat(
        [observation.unsqueeze(0), new_memory[:-1,:]], axis=0
        )

    # Update mask
    #new_mask = mask * (1 - reset)
    new_mask = mask 
    #new_mask = torch.cat([torch.ones((1)), new_mask[:-1]], axis=0)
    new_mask = torch.cat([torch.zeros((1)), new_mask[:-1]], axis=0)
    #new_mask = torch.cat([torch.zeros((1)).to(device), new_mask[:-1]], axis=0)
    return new_memory, new_mask




class PositionalEncoding_Concat(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=memory_size_read):
        super(PositionalEncoding_Concat, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        e = math.e
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        pe = (e**(-positions/100))
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = torch.cat([x,self.pe],dim=1)
        return x

class PositionalEncoding_Concat_Deco(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1):
        super(PositionalEncoding_Concat_Deco, self).__init__()

        pe2 = torch.zeros(max_len, d_model)
        e = math.e
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        pe2 = (e**(-positions/100))
        self.register_buffer('pe2', pe2)

    def forward(self, x):
        x = torch.cat([x,self.pe2],dim=1)
        return x

class PositionalEncoding_Sum(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding_Sum, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class PositionalEncoding_posi(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=memory_size_read):
        super(PositionalEncoding_posi, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        div_term = torch.exp(torch.arange(d_model,0, -2).float() * (-math.log(1000.0) / d_model))
        self.register_buffer('div_term', div_term)
        self.register_buffer('pe', pe)

    def Posi(self,action):
        x_count = 0
        y_count = 0
        count = 0
        for i in action:
            if i[2]%8==0:
                if i[0] == 1:
                    x_count += 1
                elif i[0] == -1:
                    x_count -= 1
                elif i[1] == 1:
                    y_count -= 1
                elif i[1] == -1:
                    y_count += 1
                else:
                    pass
            elif i[2]%8==1:
                if i[0] == 1:
                    x_count += 1
                    y_count += 1
                elif i[0] == -1:
                    x_count -= 1
                    y_count -= 1
                elif i[1] == 1:
                    x_count += 1
                    y_count -= 1
                elif i[1] == -1:
                    x_count -= 1
                    y_count += 1
                else:
                    pass
            elif i[2]%8==2:
                if i[0] == 1:
                    y_count += 1
                elif i[0] == -1:
                    y_count -= 1
                elif i[1] == 1:
                    x_count += 1
                elif i[1] == -1:
                    x_count -= 1
                else:
                    pass
            elif i[2]%8==3:
                if i[0] == 1:
                    x_count -= 1
                    y_count += 1
                elif i[0] == -1:
                    x_count += 1
                    y_count -= 1
                elif i[1] == 1:
                    x_count += 1
                    y_count += 1
                elif i[1] == -1:
                    x_count -= 1
                    y_count -= 1
                else:
                    pass
            elif i[2]%8==4:
                if i[0] == 1:
                    x_count -= 1
                elif i[0] == -1:
                    x_count += 1
                elif i[1] == 1:
                    y_count += 1
                elif i[1] == -1:
                    y_count -= 1
                else:
                    pass
            elif i[2]%8==5:
                if i[0] == 1:
                    x_count -= 1
                    y_count -= 1
                elif i[0] == -1:
                    x_count += 1
                    y_count += 1
                elif i[1] == 1:
                    x_count -= 1
                    y_count += 1
                elif i[1] == -1:
                    x_count += 1
                    y_count -= 1
                else:
                    pass
            elif i[2]%8==6:
                if i[0] == 1:
                    y_count -= 1
                elif i[0] == -1:
                    y_count += 1
                elif i[1] == 1:
                    x_count -= 1
                elif i[1] == -1:
                    x_count += 1
                else:
                    pass
            elif i[2]%8==7:
                if i[0] == 1:
                    x_count += 1
                    y_count -= 1
                elif i[0] == -1:
                    x_count -= 1
                    y_count += 1
                elif i[1] == 1:
                    x_count -= 1
                    y_count -= 1
                elif i[1] == -1:
                    x_count += 1
                    y_count += 1
                else:
                    pass
            self.pe[count, 0::2] = torch.sin(x_count * self.div_term)
            self.pe[count, 1::2] = torch.cos(y_count * self.div_term)
            count += 1
        return self.pe

    def forward(self, x, action):
        po = self.Posi(action)
        x = x + po
        return x

class gcn_transformer(nn.Module):
    """GCN implementation
    """
    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=5):
        super(gcn_transformer, self).__init__()
        self.word_embedding_size = 300
        # Observation layer
        self.fc_siemense= nn.Linear(2048, 512) #PosEnco
        self.e = math.e

        # Merge layer
        self.fc_merge = nn.Linear(1024, 300)

        #Transformer
        self.transformer_model = nn.Transformer(d_model=300,nhead=5, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=300)

        #position encoding
        self.pos_encoder = PositionalEncoding_Sum(300) #Sum
        # GCN layer
        self.gcn = GCN()

        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None

        # Convolution for similarity grid
        pooling_kernel = 2
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1.register_backward_hook(self.hook_backward)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        conv1_output = (mask_size - 3 + 1)//pooling_kernel
        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        self.flat_input = 16 * conv2_output * conv2_output

        # context Merge layer 
        self.fc_merge_context = nn.Linear(
        300+self.flat_input, 300)


    def forward(self, inp):
        # x is the observation (resnet feature stacked)
        # y is the target
        # z is ContextGrid
        # o is the observation (RGB frame)

        (x, y, z, o, memory, mask, gcn_memory, gcn_mask, device) = inp

        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        emmemory = memory

        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)

        word2vec = y
        word2vec=word2vec.view(1,1,-1)

        z = torch.autograd.Variable(z, requires_grad=True)
        z = self.conv1(z)
        z.register_hook(self.save_gradient)
        self.conv_output = z
        z = self.pool(F.relu(z))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.view(-1)
        self.output_context = z

        o = o.view(3, 300, 400)
        gcn_memory = gcn_memory.to(device)
        gcn_memory, gcn_mask = update_gcn_memory(o, gcn_memory, gcn_mask, device)
        gcn_amask = torch.tensor(gcn_mask.clone().detach(), dtype=bool).to(device)
        gcn_emmemory = gcn_memory
        gcn_em_memory = torch.zeros(i,512).to(device)
        for l in range(i):
            gcn_em_memory[l] = self.gcn(gcn_emmemory[l][:][:][:].view(1, 3, 300, 400)).view(1, -1)

        k = torch.cat([j, gcn_em_memory], dim=1)
        k = F.relu(self.fc_merge(k), True)
        k=torch.cat([k,torch.zeros(memory_size_read-i,300).to(device)])
        k = self.pos_encoder(k)
        k = k.view(memory_size_read, 1, -1)

        xy = self.transformer_model(k, word2vec, src_key_padding_mask = amask, memory_key_padding_mask = amask)

        xy = xy.view(-1)
        xy = torch.cat([xy, z])
        xy = self.fc_merge_context(xy)
        xy = F.relu(xy, True)
        xy = xy.view(1, 1, -1)
        return (xy, memory, mask, gcn_memory, gcn_mask)


# Code borrowed from https://github.com/tkipf/pygcn
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


# Code borrowed from https://github.com/allenai/savn/
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        self.resnet50 = models.resnet50(pretrained=True)
        for p in self.resnet50.parameters():
            p.requires_grad = False
        self.resnet50.eval()

        # Load adj matrix for GCN
        A_raw = torch.load("./data/gcn/adjmat.dat")
        A = normalize_adj(A_raw).tocsr().toarray()
        self.A = torch.nn.Parameter(torch.Tensor(A))

        objects = open("./data/gcn/objects.txt").readlines()
        objects = [o.strip() for o in objects]
        self.n = len(objects)
        self.register_buffer('all_glove', torch.zeros(self.n, 300))

        # Every dataset contain the same word embedding use FloorPlan1
        h5_file = h5py.File("./data/FloorPlan1.h5", 'r')
        object_ids = json.loads(h5_file.attrs['object_ids'])
        object_vector = h5_file['object_vector']

        word_embedding = {k: object_vector[v] for k, v in object_ids.items()}
        for i, o in enumerate(objects):
            self.all_glove[i, :] = torch.from_numpy(word_embedding[o])

        h5_file.close()

        nhid = 1024
        # Convert word embedding to input for gcn
        self.word_to_gcn = nn.Linear(300, 512)

        # Convert resnet feature to input for gcn
        self.resnet_to_gcn = nn.Linear(1000, 512)

        # GCN net
        self.gc1 = GraphConvolution(512 + 512, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, 1)

        self.mapping = nn.Linear(self.n, 512)

    def gcn_embed(self, x):

        resnet_score = self.resnet50(x)
        resnet_embed = self.resnet_to_gcn(resnet_score)
        word_embedding = self.word_to_gcn(self.all_glove)

        output = torch.cat(
            (resnet_embed.repeat(self.n, 1), word_embedding), dim=1)
        return output

    def forward(self, x):
        #print(x.size())
        #x = x.view(1, 3, 300, 400)
        # x = (current_obs)
        # Convert input to gcn input
        x = self.gcn_embed(x)

        x = F.relu(self.gc1(x, self.A))
        x = F.relu(self.gc2(x, self.A))
        x = F.relu(self.gc3(x, self.A))
        x = x.view(-1)
        x = self.mapping(x)
        return x


class Grid_memory(nn.Module):
    """GCN implementation
    """
    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=5):
        super(Grid_memory, self).__init__()
        self.word_embedding_size = 300
        # Observation layer
        #$$$$$$$$$$$
        #self.fc_siemense= nn.Linear(2049, 512) #PosEnco
        self.fc_siemense= nn.Linear(2048, 512) #PosEnco
        #$$$$$$$$$$$
        self.e = math.e

        # Merge layer
        #$$$$$$$$$$$
        self.fc_merge = nn.Linear(912, 300)
        #no marge $$$$$$$$$$$
        #self.fc_word = nn.Linear(300, 912)
        #no marge $$$$$$$$$$$
        #self.fc_grid = nn.Linear(400, 300)
        #$$$$$$$$$$$

        #Transformer
        self.transformer_model = TF.Transformer(d_model=300,nhead=5, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=300)
        #self.fc_1 = nn.Linear(1200, 600)
        #self.fc_2 = nn.Linear(600, 300)
        #self.fc_1 = nn.Linear(9600, 600)
        #self.fc_2 = nn.Linear(600, 300)

        #position encoding
        self.pos_encoder = PositionalEncoding_Sum(300) #Sum
        #self.pos_encoder = PositionalEncoding_Concat(1) #Concat
        # GCN layer

        # Convolution for similarity grid
        #pooling_kernel = 2
        #self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        #self.conv1.register_backward_hook(self.hook_backward)
        #self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        #self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        #conv1_output = (mask_size - 3 + 1)//pooling_kernel
        #conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        #self.flat_input = 16 * conv2_output * conv2_output

        # Convolution for similarity grid2 
        pooling_kernel_2 = 2
        self.conv1_2 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1_2.register_backward_hook(self.hook_backward)
        self.pool_2 = nn.MaxPool2d(pooling_kernel_2, pooling_kernel_2)
        self.conv2_2 = nn.Conv2d(8, 16, 5, stride=1)

        # context Merge layer 
        #self.fc_merge_context = nn.Linear(812, 300)

        #Dropout
        #self.dropout = nn.Dropout(p=0.2)

    def forward(self, inp):
        # x is the observation (resnet feature stacked)
        # z is ContextGrid

        (x, z, memory, mask, word2vec, grid_memory, grid_mask, device) = inp

        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        emmemory = memory
        ###Concat####
        #emmemory = self.pos_encoder(emmemory)
        
        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)
        #j = self.dropout(F.relu(self.fc_siemense(emmemory[0:i]),True)).view(i,-1)


        z = z.view(16, 16)
        z = torch.autograd.Variable(z, requires_grad=True)
        ###
        grid_memory = grid_memory.to(device)
        grid_memory, grid_mask = update_grid_memory(z, grid_memory, grid_mask, device)
        grid_amask = torch.tensor(grid_mask.clone().detach(), dtype=bool).to(device)
        grid_emmemory = grid_memory
        grid_em_memory = torch.zeros(i,1, 8, 14, 14).to(device)
        for l in range(i):
            grid_em_memory[l] = self.conv1_2(grid_emmemory[l].view(1, 1, 16, 16))
        grid_em_memory.register_hook(self.save_gradient)
        self.conv_output = grid_em_memory
        grid_em_memory2 = torch.zeros(i, 8, 7, 7).to(device)
        grid_em_memory3 = torch.zeros(i, 400).to(device)
        for l in range(i):
            grid_em_memory2[l] = self.pool_2(F.relu(grid_em_memory[l]))
            grid_em_memory3[l] = self.pool_2(F.relu(self.conv2_2(grid_em_memory[l]))).view(-1)
        ###

        #$$$$$$$$$$$
        k = torch.cat([j, grid_em_memory3], dim=1)
        k = F.relu(self.fc_merge(k), True)
        k = torch.cat([k,torch.zeros(memory_size_read-i,300).to(device)])

        word2vec = torch.from_numpy(word2vec).to(device)
        word2vec=word2vec.view(1,1,-1)

        #no marge $$$$$$$$$$$
        #k = torch.cat([k,torch.zeros(16-i,912).to(device)])
        #word2vec = F.relu(self.fc_word(word2vec), True)
        #word2vec = word2vec.view(1,1,-1)
        #no marge $$$$$$$$$$$

        #only obs. $$$$$$$$$$$
        #k = grid_em_memory3
        #k = F.relu(self.fc_grid(k), True)
        #k = torch.cat([k,torch.zeros(memory_size_read-i,300).to(device)])
        #$$$$$$$$$$$
        k = self.pos_encoder(k)
        k = k.view(memory_size_read, 1, -1)


        #only obs. $$$$$$$$$$$
        #x = x.view(-1)
        #x = self.fc_siemense(x)
        #x = F.relu(x, True)

        xy, encoder_atten_weights, decoder_atten_weights = self.transformer_model(k, word2vec, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        ###no trans####
        #k = k.view(-1)
        #xy = self.fc_1(k)
        #xy = F.relu(xy, True)
        #xy = self.fc_2(xy)
        #xy = F.relu(xy, True)
        #encoder_atten_weights =torch.zeros(1).to(device)
        #decoder_atten_weights = torch.zeros(1).to(device)

        #only obs. $$$$$$$$$$$
        #xy = xy.view(-1)
        #xy = torch.cat([xy, x])
        #xy = self.fc_merge_context(xy)
        #xy = F.relu(xy, True)
        #xy = xy.view(1, 1, -1)
        return (xy, memory, mask, grid_memory, grid_mask, encoder_atten_weights, decoder_atten_weights)


class Scene_only(nn.Module):
    """GCN implementation
    """
    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=5):
        super(Scene_only, self).__init__()
        self.word_embedding_size = 300
        # Observation layer
        #$$$$$$$$$$$
        #self.fc_siemense= nn.Linear(2049, 512) #PosEnco
        self.fc_siemense= nn.Linear(2048, 512) #PosEnco
        #$$$$$$$$$$$
        self.e = math.e

        # Merge layer
        #$$$$$$$$$$$
        self.fc_merge = nn.Linear(912, 300)
        #no marge $$$$$$$$$$$
        #self.fc_word = nn.Linear(300, 912)
        #no marge $$$$$$$$$$$
        #self.fc_grid = nn.Linear(400, 300)
        #$$$$$$$$$$$

        #Transformer
        self.transformer_model = TF.Transformer(d_model=300,nhead=5, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=300)
        #self.fc_1 = nn.Linear(1200, 600)
        #self.fc_2 = nn.Linear(600, 300)
        #self.fc_1 = nn.Linear(9600, 600)
        #self.fc_2 = nn.Linear(600, 300)

        #position encoding
        self.pos_encoder = PositionalEncoding_Sum(300) #Sum
        #self.pos_encoder = PositionalEncoding_Concat(1) #Concat
        # GCN layer

        # Convolution for similarity grid
        #pooling_kernel = 2
        #self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        #self.conv1.register_backward_hook(self.hook_backward)
        #self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        #self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        #conv1_output = (mask_size - 3 + 1)//pooling_kernel
        #conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        #self.flat_input = 16 * conv2_output * conv2_output

        # Convolution for similarity grid2 
        pooling_kernel_2 = 2
        self.conv1_2 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1_2.register_backward_hook(self.hook_backward)
        self.pool_2 = nn.MaxPool2d(pooling_kernel_2, pooling_kernel_2)
        self.conv2_2 = nn.Conv2d(8, 16, 5, stride=1)

        # context Merge layer 
        #self.fc_merge_context = nn.Linear(812, 300)

        #Dropout
        #self.dropout = nn.Dropout(p=0.2)

    def forward(self, inp):
        # x is the observation (resnet feature stacked)
        # z is ContextGrid

        (x, z, memory, mask, word2vec, grid_memory, grid_mask, device) = inp

        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        emmemory = memory
        ###Concat####
        #emmemory = self.pos_encoder(emmemory)
        
        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)
        #j = self.dropout(F.relu(self.fc_siemense(emmemory[0:i]),True)).view(i,-1)


        z = z.view(16, 16)
        z = torch.autograd.Variable(z, requires_grad=True)
        ###
        grid_memory = grid_memory.to(device)
        grid_memory, grid_mask = update_grid_memory(z, grid_memory, grid_mask, device)
        grid_amask = torch.tensor(grid_mask.clone().detach(), dtype=bool).to(device)
        grid_emmemory = grid_memory
        grid_em_memory = torch.zeros(i,1, 8, 14, 14).to(device)
        for l in range(i):
            grid_em_memory[l] = self.conv1_2(grid_emmemory[l].view(1, 1, 16, 16))
        grid_em_memory.register_hook(self.save_gradient)
        self.conv_output = grid_em_memory
        grid_em_memory2 = torch.zeros(i, 8, 7, 7).to(device)
        grid_em_memory3 = torch.zeros(i, 400).to(device)
        for l in range(i):
            grid_em_memory2[l] = self.pool_2(F.relu(grid_em_memory[l]))
            grid_em_memory3[l] = self.pool_2(F.relu(self.conv2_2(grid_em_memory[l]))).view(-1)
            grid_em_memory3[l] = torch.zeros_like(grid_em_memory3[l]) #add for scene_only
        ###

        #$$$$$$$$$$$
        k = torch.cat([j, grid_em_memory3], dim=1)
        k = F.relu(self.fc_merge(k), True)
        k = torch.cat([k,torch.zeros(memory_size_read-i,300).to(device)])

        word2vec = torch.from_numpy(word2vec).to(device)
        word2vec=word2vec.view(1,1,-1)

        #no marge $$$$$$$$$$$
        #k = torch.cat([k,torch.zeros(16-i,912).to(device)])
        #word2vec = F.relu(self.fc_word(word2vec), True)
        #word2vec = word2vec.view(1,1,-1)
        #no marge $$$$$$$$$$$

        #only obs. $$$$$$$$$$$
        #k = grid_em_memory3
        #k = F.relu(self.fc_grid(k), True)
        #k = torch.cat([k,torch.zeros(memory_size_read-i,300).to(device)])
        #$$$$$$$$$$$
        k = self.pos_encoder(k)
        k = k.view(memory_size_read, 1, -1)


        #only obs. $$$$$$$$$$$
        #x = x.view(-1)
        #x = self.fc_siemense(x)
        #x = F.relu(x, True)

        xy, encoder_atten_weights, decoder_atten_weights = self.transformer_model(k, word2vec, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        ###no trans####
        #k = k.view(-1)
        #xy = self.fc_1(k)
        #xy = F.relu(xy, True)
        #xy = self.fc_2(xy)
        #xy = F.relu(xy, True)
        #encoder_atten_weights =torch.zeros(1).to(device)
        #decoder_atten_weights = torch.zeros(1).to(device)

        #only obs. $$$$$$$$$$$
        #xy = xy.view(-1)
        #xy = torch.cat([xy, x])
        #xy = self.fc_merge_context(xy)
        #xy = F.relu(xy, True)
        #xy = xy.view(1, 1, -1)
        return (xy, memory, mask, grid_memory, grid_mask, encoder_atten_weights, decoder_atten_weights)

class Grid_memory_no_observation(nn.Module):
    """GCN implementation
    """
    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=5):
        super(Grid_memory_no_observation, self).__init__()
        self.word_embedding_size = 300
        # Observation layer
        #$$$$$$$$$$$
        #self.fc_siemense= nn.Linear(2049, 512) #PosEnco
        self.fc_siemense= nn.Linear(2048, 512) #PosEnco
        #$$$$$$$$$$$
        self.e = math.e

        # Merge layer
        #$$$$$$$$$$$
        self.fc_merge = nn.Linear(912, 300)
        #no marge $$$$$$$$$$$
        #self.fc_word = nn.Linear(300, 912)
        #no marge $$$$$$$$$$$
        #self.fc_grid = nn.Linear(400, 300)
        #$$$$$$$$$$$

        #Transformer
        self.transformer_model = TF.Transformer(d_model=300,nhead=5, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=300)
        #self.fc_1 = nn.Linear(1200, 600)
        #self.fc_2 = nn.Linear(600, 300)
        #self.fc_1 = nn.Linear(9600, 600)
        #self.fc_2 = nn.Linear(600, 300)

        #position encoding
        self.pos_encoder = PositionalEncoding_Sum(300) #Sum
        #self.pos_encoder = PositionalEncoding_Concat(1) #Concat
        # GCN layer

        # Convolution for similarity grid
        #pooling_kernel = 2
        #self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        #self.conv1.register_backward_hook(self.hook_backward)
        #self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        #self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        #conv1_output = (mask_size - 3 + 1)//pooling_kernel
        #conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        #self.flat_input = 16 * conv2_output * conv2_output

        # Convolution for similarity grid2 
        pooling_kernel_2 = 2
        self.conv1_2 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1_2.register_backward_hook(self.hook_backward)
        self.pool_2 = nn.MaxPool2d(pooling_kernel_2, pooling_kernel_2)
        self.conv2_2 = nn.Conv2d(8, 16, 5, stride=1)

        # context Merge layer 
        #self.fc_merge_context = nn.Linear(812, 300)

        #Dropout
        #self.dropout = nn.Dropout(p=0.2)

    def forward(self, inp):
        # x is the observation (resnet feature stacked)
        # z is ContextGrid

        (x, z, memory, mask, word2vec, grid_memory, grid_mask, device) = inp #origin
        # (z, mask, word2vec, grid_memory, grid_mask, device) = inp #add

        x = x.view(-1) #origin
        # x = torch.zeros(2048).view(-1) #except for resnet
        # print("(network) zero over ride") #test
        x = torch.zeros_like(x) #except for resnet
        x = x.to(device) #except for resnet
        memory = memory.to(device)
        # print("(network) memory : {}".format(memory)) #test

        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        memory = torch.zeros_like(memory) #except for resnet
        # print("(network) memory : {}".format(memory)) #test
        emmemory = memory
        ###Concat####
        #emmemory = self.pos_encoder(emmemory) 
        
        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)
        # print("(network) j : {}".format(j)) #test
        #j = self.dropout(F.relu(self.fc_siemense(emmemory[0:i]),True)).view(i,-1)


        z = z.view(16, 16)
        z = torch.autograd.Variable(z, requires_grad=True)
        ###
        grid_memory = grid_memory.to(device)
        grid_memory, grid_mask = update_grid_memory(z, grid_memory, grid_mask, device)
        grid_amask = torch.tensor(grid_mask.clone().detach(), dtype=bool).to(device)
        grid_emmemory = grid_memory
        grid_em_memory = torch.zeros(i,1, 8, 14, 14).to(device)
        for l in range(i):
            grid_em_memory[l] = self.conv1_2(grid_emmemory[l].view(1, 1, 16, 16))
        grid_em_memory.register_hook(self.save_gradient)
        self.conv_output = grid_em_memory
        grid_em_memory2 = torch.zeros(i, 8, 7, 7).to(device)
        grid_em_memory3 = torch.zeros(i, 400).to(device)
        for l in range(i):
            grid_em_memory2[l] = self.pool_2(F.relu(grid_em_memory[l]))
            grid_em_memory3[l] = self.pool_2(F.relu(self.conv2_2(grid_em_memory[l]))).view(-1)
        ###

        #$$$$$$$$$$$
        # print("(network) j : {}".format(j)) #test
        k = torch.cat([j, grid_em_memory3], dim=1) #origin
        # k = grid_em_memory3 #except for resnet
        k = F.relu(self.fc_merge(k), True)
        k = torch.cat([k,torch.zeros(memory_size_read-i,300).to(device)])

        word2vec = torch.from_numpy(word2vec).to(device)
        word2vec=word2vec.view(1,1,-1)

        #no marge $$$$$$$$$$$
        #k = torch.cat([k,torch.zeros(16-i,912).to(device)])
        #word2vec = F.relu(self.fc_word(word2vec), True)
        #word2vec = word2vec.view(1,1,-1)
        #no marge $$$$$$$$$$$

        #only obs. $$$$$$$$$$$
        #k = grid_em_memory3
        #k = F.relu(self.fc_grid(k), True)
        #k = torch.cat([k,torch.zeros(memory_size_read-i,300).to(device)])
        #$$$$$$$$$$$
        k = self.pos_encoder(k)
        k = k.view(memory_size_read, 1, -1)


        #only obs. $$$$$$$$$$$
        #x = x.view(-1)
        #x = self.fc_siemense(x)
        #x = F.relu(x, True)

        xy, encoder_atten_weights, decoder_atten_weights = self.transformer_model(k, word2vec, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        ###no trans####
        #k = k.view(-1)
        #xy = self.fc_1(k)
        #xy = F.relu(xy, True)
        #xy = self.fc_2(xy)
        #xy = F.relu(xy, True)
        #encoder_atten_weights =torch.zeros(1).to(device)
        #decoder_atten_weights = torch.zeros(1).to(device)

        #only obs. $$$$$$$$$$$
        #xy = xy.view(-1)
        #xy = torch.cat([xy, x])
        #xy = self.fc_merge_context(xy)
        #xy = F.relu(xy, True)
        #xy = xy.view(1, 1, -1)
        return (xy, memory, mask, grid_memory, grid_mask, encoder_atten_weights, decoder_atten_weights)


def update_grid_memory(observation, memory, mask, device):
    """
    Update the memory and mask based on latest observation
    """
    assert (
        observation.shape[0] == memory.shape[1]
    ), f"Embedding sizes don't match, {observation.shape[0]} vs {memory.shape[1]}"
    assert mask.shape[0] == memory.shape[0], f"Memory sizes don't match"
    #assert len(reset.shape.as_list()) == 0, f"Reset must be scalar"

    new_memory = memory 
    # Shift memory forward and add new observation
    new_memory = torch.cat(
        [observation.unsqueeze(0), new_memory[:-1,:]], axis=0
        )

    # Update mask
    #new_mask = mask * (1 - reset)
    new_mask = mask 
    #new_mask = torch.cat([torch.ones((1)), new_mask[:-1]], axis=0)
    new_mask = torch.cat([torch.zeros((1)), new_mask[:-1]], axis=0)
    #new_mask = torch.cat([torch.zeros((1)).to(device), new_mask[:-1]], axis=0)
    return new_memory, new_mask


class Grid_memory_action(nn.Module):
    """GCN implementation
    """
    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=5):
        super(Grid_memory_action, self).__init__()
        self.word_embedding_size = 300
        # Observation layer
        #$$$$$$$$$$$
        self.fc_siemense= nn.Linear(2048, 512) #PosEnco
        self.fc_action= nn.Linear(128, 256) #PosEnco
        #self.fc_action= nn.Linear(memory_size_read, 64) #PosEnco
        #$$$$$$$$$$$
        self.e = math.e

        # Merge layer
        #####Trans#####
        #self.fc_merge = nn.Linear(976, 300)
        self.fc_merge = nn.Linear(912, 300)
        self.fc_merge_ac = nn.Linear(768, 512)
        self.fc_merge_last = nn.Linear(812, 300)
        self.transformer_model_ac = TF.Transformer(d_model=512,nhead=8, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=512)
        #####Trans#####
        #Transformer
        self.transformer_model = TF.Transformer(d_model=300,nhead=5, num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=300)

        #position encoding
        self.pos_encoder = PositionalEncoding_Sum(300) #Sum
        self.pos_encoder_ac = PositionalEncoding_Sum(512) #Sum
        # GCN layer

        # Convolution for similarity grid2 
        pooling_kernel_2 = 2
        self.conv1_2 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1_2.register_backward_hook(self.hook_backward)
        self.pool_2 = nn.MaxPool2d(pooling_kernel_2, pooling_kernel_2)
        self.conv2_2 = nn.Conv2d(8, 16, 5, stride=1)

        # context Merge layer 
        #self.fc_merge_context = nn.Linear(300+self.flat_input, 300)

        #Dropout
        #self.dropout = nn.Dropout(p=0.5)

    def forward(self, inp):
        # x is the observation (resnet feature stacked)
        # z is ContextGrid

        (x, z, memory, mask, word2vec, grid_memory, grid_mask, last_act, act_memory, act_mask, device) = inp

        #$$$$$$$$$$$
        x = x.view(-1)
        memory = memory.to(device)
        memory, mask = update_memory(x, memory, mask,device)
        mask = mask.view(1,-1)
        amask = torch.tensor(mask.clone().detach(), dtype=bool).to(device)
        emmemory = memory

        i = (mask[0] == 0).nonzero()[-1].numpy()[-1] + 1
        j = F.relu(self.fc_siemense(emmemory[0:i]),True).view(i,-1)
        #j = self.dropout(F.relu(self.fc_siemense(emmemory[0:i]),True)).view(i,-1)
        #$$$$$$$$$$$

        word2vec = torch.from_numpy(word2vec).to(device)
        word2vec=word2vec.view(1,1,-1)

        last_act.view(-1)
        act_memory = act_memory.to(device)
        act_memory, act_mask = update_memory_action(last_act, act_memory, act_mask,device)
        act_mask = act_mask.view(1,-1)
        act_amask = torch.tensor(act_mask.clone().detach(), dtype=bool).to(device)
        act_emmemory = act_memory

        ac = F.relu(self.fc_action(act_emmemory[0:i]),True).view(i,-1)

        ###
        z = z.view(16, 16)
        z = torch.autograd.Variable(z, requires_grad=True)
        ###
        grid_memory = grid_memory.to(device)
        grid_memory, grid_mask = update_grid_memory(z, grid_memory, grid_mask, device)
        grid_amask = torch.tensor(grid_mask.clone().detach(), dtype=bool).to(device)
        grid_emmemory = grid_memory
        grid_em_memory = torch.zeros(i,1, 8, 14, 14).to(device)
        for l in range(i):
            grid_em_memory[l] = self.conv1_2(grid_emmemory[l].view(1, 1, 16, 16))
        ###
        grid_em_memory.register_hook(self.save_gradient)
        self.conv_output = grid_em_memory
        grid_em_memory2 = torch.zeros(i, 8, 7, 7).to(device)
        grid_em_memory3 = torch.zeros(i, 400).to(device)
        for l in range(i):
            grid_em_memory2[l] = self.pool_2(F.relu(grid_em_memory[l]))
            grid_em_memory3[l] = self.pool_2(F.relu(self.conv2_2(grid_em_memory[l]))).view(-1)
        ###
        #$$$$$$$$$$$
        #k = torch.cat([j, grid_em_memory3], dim=1)
        #k = torch.cat([k, ac], dim=1)
        #k = F.relu(self.fc_merge(k), True)
        #k = torch.cat([k,torch.zeros(16-i,300).to(device)])
        #k = self.pos_encoder(k)
        #k = k.view(16, 1, -1)
        #$$$$$$$$$$$
        ####Trans####
        k = torch.cat([j, grid_em_memory3], dim=1)
        k = F.relu(self.fc_merge(k), True)
        k = torch.cat([k,torch.zeros(memory_size_read-i,300).to(device)])
        k = self.pos_encoder(k)
        k = k.view(memory_size_read, 1, -1)

        m = torch.cat([j, ac], dim=1)
        m = F.relu(self.fc_merge_ac(m), True)
        m = torch.cat([m,torch.zeros(memory_size_read-i,512).to(device)])
        m = self.pos_encoder_ac(m)
        m = m.view(memory_size_read, 1, -1)

        x = x.view(-1)
        x = self.fc_siemense(x)
        x = F.relu(x, True)
        x = x.view(1, 1, -1)
        ####Trans####
        xy_ac, encoder_atten_weights_ac, decoder_atten_weights_ac = self.transformer_model_ac(m, x, src_key_padding_mask = amask, memory_key_padding_mask = amask)
        ####Trans####
        xy, encoder_atten_weights, decoder_atten_weights = self.transformer_model(k, word2vec, src_key_padding_mask = amask, memory_key_padding_mask = amask)

        xy = xy.view(-1)
        xy_ac = xy_ac.view(-1)
        xyz = torch.cat([xy, xy_ac])
        xyz = self.fc_merge_last(xyz)
        xyz = F.relu(xyz, True)
        xyz = xyz.view(1, 1, -1)
        return (xyz, memory, mask, grid_memory, grid_mask, act_memory, act_mask, encoder_atten_weights, decoder_atten_weights)
        ####Trans####
        #return (xy, memory, mask, grid_memory, grid_mask, act_memory, act_mask, encoder_atten_weights, decoder_atten_weights)

    
