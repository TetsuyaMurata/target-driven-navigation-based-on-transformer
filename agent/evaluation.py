from tkinter import W
from agent.network import SharedNetwork, SceneSpecificNetwork
from agent.environment import THORDiscreteEnvironment
#from agent.training import TrainingSaver, TOTAL_PROCESSED_FRAMES
from agent.training import TrainingSaver
from agent.utils import find_restore_point
import torch.nn.functional as F
import torch
import pickle
import os
import numpy as np
import re
from itertools import groupby
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import csv
from _ctypes import PyObj_FromPtr
import math
import pathlib #ADD
import glob #ADD

with open('.env', mode='r', encoding='utf-8') as f:
    target_path = "EXPERIMENT/" + f.readline().replace('\n', '')
print("TARGET : {}".format(target_path.replace("EXPERIMENT/", "")))

# for making videos by directories
os.makedirs("video/" + target_path.replace("EXPERIMENT/", ""), exist_ok=True)

with open('.target_path', mode='w', encoding='utf-8') as f:
    f.write(target_path)

with open(".error_point", mode="w", encoding="utf-8") as f_error:
    f_error.write("")

json_open = open(target_path + "/"+ "param.json", "r")
json_load = json.load(json_open)

memory_size_read = json_load['memory']
print("(evaluation) memory_size : {}".format(str(memory_size_read))) #test

output_log_path = target_path

count_files = len(glob.glob("./tmp/*"))
# ###### ADD ######

#ACTION_SPACE_SIZE = 4
NUM_EVAL_EPISODES = 5
VERBOSE = True
class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError('Only lists and tuples can be wrapped')
        self.value = value

class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {'cls', 'indent'}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                    else super(MyEncoder, self).default(obj))

    def iterencode(self, obj, **kwargs):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super(MyEncoder, self).iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace(
                            '"{}"'.format(format_spec.format(id)), json_repr)

            yield encoded

def write_text(img, text, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = pos
    fontScale = 0.8
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    text_width, text_height = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, text_width, text_height

#def create_img_word2vec(target, obs, obs_feature, word_embedding, simi_grid, deco):
def create_img_word2vec(target, obs, obs_feature, simi_grid, enco, deco, time, action):
    padding = 3
    base_img = np.zeros((720, 1280, 3), dtype=np.uint8)

    """OBSERVATION
    """
    # Set obs width and height
    width_obs, height_obs = 400, 300
    #  Set obs position
    pos_width_obs, pos_height_obs = 650, 200

    obs = cv2.resize(obs, dsize=(int(width_obs*1.5), int(height_obs*1.5)))

    # Set obs width and height
    height_obs, width_obs, _ = obs.shape

    base_img, text_width, text_height = write_text(
        base_img, "Observation", (200+pos_width_obs, pos_height_obs))

    # Merge obs in canva
    base_img[pos_height_obs+text_height:pos_height_obs+height_obs+text_height+padding*2,
             pos_width_obs:pos_width_obs+width_obs+padding*2, :] = np.pad(obs, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=255)

    """OBSERVATION FEATURE
    """
    # Set observation feature position
    pos_width_feat, pos_height_feat = 100, 25

    # Normalize
    obs_feature = (obs_feature - np.min(obs_feature)) / (np.max(obs_feature) - np.min(obs_feature))
    obs_feature = obs_feature * 255

    # Vector to matrix and upscale
    obs_feature = obs_feature.reshape(
        (32, -1, 1)).repeat(3, axis=2).repeat(2, axis=0).repeat(2, axis=1)

    base_img, text_width, text_height = write_text(
        base_img, "Visual feature", (pos_width_feat, pos_height_feat))

    # Get obs width and height
    height_feat, width_feat, _ = obs_feature.shape

    base_img[pos_height_feat+text_height:pos_height_feat+text_height+height_feat+padding*2,
             pos_width_feat:pos_width_feat+width_feat+padding*2, :] = np.pad(obs_feature, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=255)

    """WORD EMBEDDING FEATURE
    """
#    # Set observation feature position
#    pos_width_we, pos_height_we = 100, 250
#
#    # Normalize
#    word_embedding = (word_embedding - np.min(word_embedding)) / \
#        (np.max(word_embedding) - np.min(word_embedding))
#    word_embedding = word_embedding * 255
#
#    # Vector to matrix and upscale
#    word_embedding = word_embedding.reshape(
#        (10, -1, 1)).repeat(3, axis=2).repeat(8, axis=0).repeat(8, axis=1)
#
#    base_img, text_width, text_height = write_text(
#        base_img, "Word embedding feature", (pos_width_we, pos_height_we))
#
#    # Get obs width and height
#    height_we, width_we, _ = word_embedding.shape
#
#    base_img[pos_height_we+text_height:pos_height_we+text_height+height_we+padding*2,
#             pos_width_we:pos_width_we+width_we+padding*2, :] = np.pad(word_embedding, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=255)
#
    """GRID
    """
    # Set observation feature position
    pos_width_grid, pos_height_grid = 200, 150

    # Normalize
    simi_grid = (simi_grid - np.min(simi_grid)) / \
        (np.max(simi_grid) - np.min(simi_grid))
    simi_grid = simi_grid * 255

    # Vector to matrix and upscale
    simi_grid = simi_grid.repeat(3, axis=2).repeat(12, axis=0).repeat(12, axis=1)

    base_img, text_width, text_height = write_text(
        base_img, "Similarity grid", (pos_width_grid, pos_height_grid))

    # Get obs width and height
    height_grid, width_grid, _ = simi_grid.shape

    base_img[pos_height_grid+text_height:pos_height_grid+text_height+height_grid+padding*2,
             pos_width_grid:pos_width_grid+width_grid+padding*2, :] = np.pad(simi_grid, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=255)

    """TARGET NAME
    """
    base_img, text_width, text_height = write_text(
        base_img, "TARGET : " + target, (200+pos_width_obs, pos_height_obs-150))
    """TIME
    """
    base_img, text_width, text_height = write_text(
        base_img, "TIME : " + str(time), (200+pos_width_obs, pos_height_obs-100))
    """ACTION
    """
    acts = ["MoveAhead", "RotateRight", "RotateLeft", "MoveBack",
            "LookUp", "LookDown", "MoveRight", "MoveLeft", "Done"]
    if action == " ":
        base_img, text_width, text_height = write_text(
            base_img, "TARGET : " + "Done", (200+pos_width_obs, pos_height_obs-50))
    else:
        base_img, text_width, text_height = write_text(
            base_img, "ACTION : " + acts[action], (200+pos_width_obs, pos_height_obs-50))

    """Enco
    """
    # Normalize
    #for i in range(len(enco[0])):
    #    enco[0][i] = (enco[0][i] - np.min(enco[0][i])) / \
    #        (np.max(enco[0][i]) - np.min(enco[0][i]))
    #    enco[0][i] = enco[0][i] * 255
    #print(enco)
    #enco = (enco - np.min(enco)) / \
    #    (np.max(enco) - np.min(enco))
    enco = (enco - 0) / \
        (np.max(enco) - 0)
    #print(enco)
    enco = enco * 255
    #print(enco)
    enco = np.array(enco)
    enco = enco.reshape(32, 32, 1)
    #print(enco.shape)
    # Vector to matrix and upscale
    enco = enco.repeat(3, axis=2).repeat(8, axis=0).repeat(8, axis=1)
    #print(enco.shape)
    # Set observation feature position
    pos_width_enco, pos_height_enco = 100, 400

    base_img, text_width, text_height = write_text(
        base_img, "Encoder Self Attention", (pos_width_enco, pos_height_enco))

    # Get obs width and height
    height_enco, width_enco, _ = enco.shape
    #print(enco.shape)

    base_img[pos_height_enco+text_height:pos_height_enco+text_height+height_enco+padding*2,
             pos_width_enco:pos_width_enco+width_enco+padding*2, :] = np.pad(enco, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=255)

    """Deco
    """
    # Set observation feature position
    pos_width_deco, pos_height_deco = 400, 400

    # Normalize
    deco = (deco - 0) / \
        (np.max(deco) - 0)
    deco = deco * 255
    deco = deco.reshape(-1,1,1)

    # Vector to matrix and upscale
    deco = deco.repeat(3, axis=2).repeat(8, axis=0).repeat(12, axis=1)

    base_img, text_width, text_height = write_text(
        base_img, "Decoder Attention", (pos_width_deco, pos_height_deco))

    # Get obs width and height
    height_deco, width_deco, _ = deco.shape

    base_img[pos_height_deco+text_height:pos_height_deco+text_height+height_deco+padding*2,
             pos_width_deco:pos_width_deco+width_deco+padding*2, :] = np.pad(deco, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=255)

    return base_img

def create_img(target, obs):
    padding = 3
    base_img = np.zeros((720, 1280, 3), dtype=np.uint8)

    """OBSERVATION
    """
    # Set obs width and height
    width_obs, height_obs = 400, 300
    #  Set obs position
    pos_width_obs, pos_height_obs = 100, 100

    obs = cv2.resize(obs, dsize=(int(width_obs*1.5), int(height_obs*1.5)))

    # Set obs width and height
    height_obs, width_obs, _ = obs.shape

    base_img, text_width, text_height = write_text(
        base_img, "Observation", (200+pos_width_obs, pos_height_obs))

    # Merge obs in canva
    base_img[pos_height_obs+text_height:pos_height_obs+height_obs+text_height+padding*2,
             pos_width_obs:pos_width_obs+width_obs+padding*2, :] = np.pad(obs, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=255)

    """TARGET IMAGE
    """
    # Set obs width and height
    width_target, height_target = 280, 210
    #  Set obs position
    pos_width_target, pos_height_target = 800, 100

    target = cv2.resize(target, dsize=(int(width_target*1.5), int(height_target*1.5)))

    # Set obs width and height
    height_target, width_target, _ = target.shape

    base_img, text_width, text_height = write_text(
        base_img, "TARGET IMAGE " , (100+pos_width_target, pos_height_target))

    base_img[pos_height_target+text_height:pos_height_target+height_target+text_height+padding*2,
             pos_width_target:pos_width_target+width_target+padding*2, :] = np.pad(target, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=255)

    return base_img


def export_to_csv(data, file):
    import csv
    with open(file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for k, g in groupby(sorted(data, key = lambda x: x[0]), key = lambda x: x[0]):
            g = list(g)
            header = [k, '']
            header.extend((np.mean(a) for a in list(zip(*g))[2:]))
            writer.writerow(header)
            for item in g:
                writer.writerow(list(item))
    print(f'CSV file stored "{file}"')


def get_concat_h_resize(im1, im2, resample=Image.BICUBIC, resize_big_image=True):
    if im1.height == im2.height:
        _im1 = im1
        _im2 = im2
    elif (((im1.height > im2.height) and resize_big_image) or
          ((im1.height < im2.height) and not resize_big_image)):
        _im1 = im1.resize((int(im1.width * im2.height / im1.height), im2.height), resample=resample)
        _im2 = im2
    else:
        _im1 = im1
        _im2 = im2.resize((int(im2.width * im1.height / im2.height), im1.height), resample=resample)
    dst = Image.new('RGB', (_im1.width + _im2.width, _im1.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (_im1.width, 0))
    return dst


def create_marged_view(scene_scope, name_idx, directory,start_idx,task_scope):
    filename = directory+'/'+name_idx+'.png'
    img = Image.open(filename)
    directory = './topdown_scene_view/'+str(scene_scope)+'/'+str(task_scope)+'/'
    filename = directory+str(start_idx)+'.png'
    img2 = Image.open(filename)
    print(start_idx)

    if not os.path.exists('./marged/'+str(scene_scope)):
        os.mkdir('./marged/'+str(scene_scope))
    if not os.path.exists('./marged/'+str(scene_scope)+'/'+str(task_scope)):
        os.mkdir('./marged/'+str(scene_scope)+'/'+str(task_scope))
    get_concat_h_resize(img, img2).save('./marged/'+str(scene_scope)+'/'+str(task_scope)+'/'+str(name_idx)+'.png')

class Evaluation:
    #def __init__(self, config, device, SSL, method):
    def __init__(self, config):
        self.config = config
        self.SSL = config.get('SSL')
        self.Posi = config.get('Posi')
        self.Key = config.get('Key')
        self.NGPU = config.get('NGPU')
        self.NThreads = config.get('num_thread')
        self.method = config.get('method')
        self.tasks = config.get('task_list')
        self.action_size = config.get('action_size')
        #self.device = device
        if torch.cuda.is_available() and config.get('cuda'):
            #self.cuda = True
            self.device = torch.device("cuda:0")
        else:
            #self.cuda = False
            self.device = torch.device("cpu")
        #self.method = method
        self.shared_net = SharedNetwork(self.method).to(self.device).eval()
        #self.SSL = SSL
        if self.SSL:
            #self.scene_nets = { key:SceneSpecificNetwork(ACTION_SPACE_SIZE).to(self.device) for key in TASK_LIST.keys() }
            self.scene_nets = { key:SceneSpecificNetwork(self.action_size).to(self.device) for key in TASK_LIST.keys() }
        else:
            #self.scene_nets = SceneSpecificNetwork(ACTION_SPACE_SIZE,self.method).to(self.device)
            self.scene_nets = SceneSpecificNetwork(self.action_size,self.method).to(self.device).eval()
        self.add_reward = config.get('ADDREWARD')
        # if self.method == "Transformer_Sum" or self.method =="Transformer_Concat" or "Transformer_word2vec_notarget" or self.method=="Transformer_word2vec_notarget_withposi" or self.method=="Transformer_word2vec_notarget_word2vec"or self.method=="Transformer_word2vec_notarget_word2vec_posi" or self.method=="Transformer_word2vec_notarget_word2vec_concat" or self.method=="Transformer_word2vec_notarget_action" or self.method=="Transformer_word2vec_notarget_word2vec_action" or self.method=="grid_memory" or self.method=='Transformer_word2vec_notarget_word2vec_action_posi' or self.method=="grid_memory_action": #origin
        if self.method == "Transformer_Sum" or self.method =="Transformer_Concat" or "Transformer_word2vec_notarget" or self.method=="Transformer_word2vec_notarget_withposi" or self.method=="Transformer_word2vec_notarget_word2vec"or self.method=="Transformer_word2vec_notarget_word2vec_posi" or self.method=="Transformer_word2vec_notarget_word2vec_concat" or self.method=="Transformer_word2vec_notarget_action" or self.method=="Transformer_word2vec_notarget_word2vec_action" or self.method=="grid_memory" or self.method=="grid_memory_no_observation" or self.method=='Transformer_word2vec_notarget_word2vec_action_posi' or self.method=="grid_memory_action" or self.method=="scene_only":
            ############MEMORY############
            # self.memory_size =32 #origin
            self.memory_size = memory_size_read #add
            self.embed_size = 2048
            self.batch_size = 1
    @staticmethod
    #def load_checkpoint(config, device, SSL, method, fail = True):
    def load_checkpoint(config, fail = True):
        checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        print("!!! checkpoint_path !!! : {}".format(checkpoint_path)) #ADD
        (base_name, restore_point) = find_restore_point(checkpoint_path, fail)
        print(f'Restoring from checkpoint {restore_point}')
        ###### ADD ######
        p_new = pathlib.Path(str(output_log_path) + '/eval' + str(restore_point) + '.log') #origin
        # print("$$$$$$$$$$ {} : ".format(count_files)) # count ok?
        with open("./tmp/output_log_path_{}.txt".format(count_files), mode="w", encoding="utf-8") as f:
            f.write(str(p_new))
        with p_new.open(mode='w', encoding="utf-8") as f:
            f.write('')
        ###### ADD ######
        state = torch.load(open(os.path.join(os.path.dirname(checkpoint_path), base_name), 'rb'))
        print("!!! torch load !!! : {}".format(os.path.join(os.path.dirname(checkpoint_path), base_name))) #ADD
        #evaluation = Evaluation(config,device,SSL, method)
        evaluation = Evaluation(config)
        saver = TrainingSaver(evaluation.shared_net, evaluation.scene_nets, None, evaluation.config, config.get('SSL'))
        print('Configuration')
        saver.restore(state)
        saver.print_config(offset = 4)            
        return evaluation

    def build_agent(self, scene_name):
        parent = self
        net = torch.nn.Sequential(parent.shared_net, parent.scene_nets[scene_name])
        class Agent:
            def __init__(self, initial_state, target):
                self.env = THORDiscreteEnvironment(
                    scene_name=scene_name,
                    initial_state_id = initial_state,
                    terminal_state_id = target,
                    h5_file_path=(lambda scene: parent.config["h5_file_path"].replace("{scene}", scene_name))
                )

                self.env.reset()
                self.net = net

            @staticmethod
            def get_parameters():
                return net.parameters()

            def act(self):
                with torch.no_grad():
                    state = torch.Tensor(self.env.render(mode='resnet_features')).to(parent.device)
                    target = torch.Tensor(self.env.render_target(mode='resnet_features')).to(parent.device)
                    (policy, value,) = net.forward((state, target,))
                    action = F.softmax(policy, dim=0).multinomial(1).cpu().data.numpy()[0]

                self.env.step(action)
                return (self.env.is_terminal, self.env.collided, self.env.reward)
        return Agent
        
    def save_video(self, ep_lengths, ep_actions, ep_start, env, scene_scope, task_scope):
        # Find episode based on episode length
        print("VIDEO SAVING")
        ep_lengths = np.array(ep_lengths)
        ep_actions_succeed = np.array(ep_actions)
        sorted_ep_lengths = np.sort(ep_lengths)
        ep_start_succeed = np.array(ep_start)
        ind_list = []
        name_video = []
        for idx, ep_len in enumerate(sorted_ep_lengths):
            if ep_len >= 5:
                index_best = idx
                break
        index_best = np.where(ep_lengths == sorted_ep_lengths[index_best])
        index_best = index_best[0][0]
        index_worst = np.where(ep_lengths == sorted_ep_lengths[-1])
        index_worst = index_worst[0][0]
        index_median = np.where(ep_lengths == sorted_ep_lengths[len(sorted_ep_lengths)//2])
        index_median = index_median[0][0]
        name_video = ['best', 'median', 'worst']
        ind_list = [index_best, index_median, index_worst]
        # Create dir if not exisiting
        # directory = os.path.join('./video/'+str(scene_scope)) #origin
        directory = os.path.join('./video/' + target_path.replace("EXPERIMENT/", "") + '/' + str(scene_scope)) #add

        if not os.path.exists(directory):
            os.makedirs(directory)
        #for ind in range(len(ep_lengths)):#0-179
        for idx_name, idx in enumerate(ind_list):
            # Create video to save
            #print(idx_name)
            #print(idx)
            height, width, layers = 720, 1280, 3
            filename = os.path.join(directory, 'SCENE'+'-'+str(scene_scope)+'_'+'TASKNUM'+'-'+str(task_scope)+'_'+str(name_video[idx_name]))
            video_name = os.path.join(filename + '.avi')
            text_name = os.path.join(filename + '.json')
            FPS = 5
            video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (width, height))
            # Retrieve start position
            # Set start position
            state_id_best = ep_start_succeed[idx]
            env.reset()
            env.current_state_id = state_id_best
            for a in ep_actions_succeed[idx]:
                img = create_img(env.render_target(mode='image'), env.render(mode='image'))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)
                #print(a)
                env.step(a)
           

            img = create_img(env.render_target(mode='image'), env.render(mode='image'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for i in range(10):
                video.write(img)
            video.release()
            data = {}
            data['stop'] = env.current_state_id
            def fmt(v):
                return "%.2f" % (v,)
            vecfmt = np.vectorize(fmt)
            with open(text_name, 'w') as outfile:
                json.dump(data, outfile, cls=MyEncoder, sort_keys=True, indent=4)

    def save_video_word2vec(self, ep_lengths, ep_actions, ep_start, ind_succ_or_fail_ep, env, scene_scope, task_scope, ep_enco, ep_deco,success=True): #Origin
    # def save_video_word2vec(self, ep_lengths, ep_actions, ep_start, ind_succ_or_fail_ep, env, scene_scope, task_scope, ep_enco, ep_deco,success=False):
        # Find episode based on episode length
        if not ind_succ_or_fail_ep:
            return
        ep_lengths = np.array(ep_lengths)
        sorted_ep_lengths = np.sort(ep_lengths[ind_succ_or_fail_ep])
        ep_lengths_succeed = ep_lengths[ind_succ_or_fail_ep]
        ep_actions_succeed = np.array(ep_actions)[ind_succ_or_fail_ep]
        ep_start_succeed = np.array(ep_start)[ind_succ_or_fail_ep]
        ep_enco = np.array(ep_enco)[ind_succ_or_fail_ep]
        ep_deco = np.array(ep_deco)[ind_succ_or_fail_ep]

        ind_list = []
        names_video = []
        if success:
            # Best is the first episode in the sorted list but we want more than 5 step
            index_best = 0
            for idx, ep_len in enumerate(sorted_ep_lengths):
                if ep_len >= 5:
                    index_best = idx
                    break
            index_best = np.where(
                ep_lengths_succeed == sorted_ep_lengths[index_best])
            index_best = index_best[0][0]
            # print("Best", ep_lengths_succeed[index_best])

            # Worst is the last episode in the sorted list
            index_worst = np.where(
                ep_lengths_succeed == sorted_ep_lengths[-1])
            index_worst = index_worst[0][0]
            # print("Worst", ep_lengths_succeed[index_worst])

            # Median is half the array size
            index_median = np.where(
                ep_lengths_succeed == sorted_ep_lengths[len(sorted_ep_lengths)//2])
            # Extract index
            index_median = index_median[0][0]
            # print("Median", ep_lengths_succeed[index_median])
            names_video = ['best', 'median', 'worst']
            ind_list = [index_best, index_median, index_worst]
        else:
            ind_list = [i for i in range(len(ind_succ_or_fail_ep))]
            names_video = ['Fail_' + str(i) for i in range(len(ind_succ_or_fail_ep))]
            print("names_video : {}".format(names_video)) #add

        # Create dir if not exisiting
        # directory = os.path.join('./video/'+str(scene_scope)) #origin
        directory = os.path.join('./video/' + target_path.replace("EXPERIMENT/", "") + '/' + str(scene_scope)) #add
        if not os.path.exists(directory):
            os.makedirs(directory)
        for idx_name, idx in enumerate(ind_list):
            # Create video to save
            height, width, layers = 720, 1280, 3
            filename = os.path.join(directory, scene_scope + '_' +
                                        task_scope['object'] + '_' +
                                        names_video[idx_name] + '_' +
                                        str(ep_lengths_succeed[idx]))
            video_name = os.path.join(filename + '.avi')
            text_name = os.path.join(filename + '.json')
            FPS = 5
            # print("(evaluation) video output") #test
            video = cv2.VideoWriter(
                video_name, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (width, height))
            # Retrieve start position
            state_id_best = ep_start_succeed[idx]
            env.reset()

            # Set start position
            env.current_state_id = state_id_best
            time = 0
            for a in ep_actions_succeed[idx]:
                state, x_processed, object_mask = self.extract_input(env, torch.device("cpu"))
                #state, x_processed, object_mask, hidden = self.method_class.extract_input(env, torch.device("cpu"))
                x_processed = x_processed.view(-1, 1).numpy()
                object_mask = object_mask.squeeze().unsqueeze(2).numpy()
                object_mask = np.flip(np.rot90(object_mask), axis=0)
                #img = create_img_word2vec(task_scope['object'], env.render(mode='image'), x_processed,
                #                    np.zeros((300, 1)), object_mask)
                if time == len(ep_actions_succeed[idx]) - 1:
                    img = create_img_word2vec(task_scope['object'], env.render(mode='image'), x_processed,
                                        object_mask, ep_enco[idx][time], ep_deco[idx][time], time, " ")
                else:
                    img = create_img_word2vec(task_scope['object'], env.render(mode='image'), x_processed,
                                        object_mask, ep_enco[idx][time], ep_deco[idx][time], time, ep_actions_succeed[idx][time])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)
                env.step(a)
                time += 1
            for i in range(10):
                video.write(img)
            video.release()

            data = {}
            data['start'] = ep_start_succeed[idx]
            data['stop'] = env.current_state_id
            data['action'] = [env.acts[i] for i in ep_actions_succeed[idx]]
            data['object_visible'] = [k.split("|")[0] for k in env.boudingbox.keys()]
            round_mask = np.squeeze(np.around(object_mask,2)).tolist()

            def fmt(v):
                return "%.2f" % (v,)
            vecfmt = np.vectorize(fmt)

            data['object_mask'] = [NoIndent(e) for e in vecfmt(round_mask).tolist()]

            with open(text_name, 'w') as outfile:
                json.dump(data, outfile, cls=MyEncoder, sort_keys=True, indent=4)


    def create_topdown(self,ep_lengths, ep_start,ep_pos_x,ep_pos_y, ep_pos_xy, scene_scope,task_scope, ep_target):
        print("Create topdown")
        ep_lengths = np.array(ep_lengths)
        sorted_ep_lengths = np.sort(ep_lengths)
        ind_list = []
        name_video = []
        for idx, ep_len in enumerate(sorted_ep_lengths):
            if ep_len >= 5:
                index_best = idx
                break
        index_best = np.where(ep_lengths == sorted_ep_lengths[index_best])
        index_best = index_best[0][0]
        index_worst = np.where(ep_lengths == sorted_ep_lengths[-1])
        index_worst = index_worst[0][0]
        index_median = np.where(ep_lengths == sorted_ep_lengths[len(sorted_ep_lengths)//2])
        index_median = index_median[0][0]
        name_video = ['best', 'median', 'worst']
        ind_list = [index_best, index_median, index_worst]
        directory = os.path.join('./topdown/'+str(scene_scope)+'/'+str(task_scope))
        if not os.path.exists(directory):
            os.makedirs(directory)
        num = 0
        for idx_name, idx in enumerate(ind_list):
            #if num == 1:
            fig = plt.figure()
            ax = fig.add_subplot()
            #ax.set_xlim(-5,5)
            #ax.set_ylim(-5,5)
            ax.set_xlim(0,10)
            ax.set_ylim(-10,0)
            R = 'ff'
            G = '00'
            B = '00'
            color_code = ['00','11','22','33','44','55','66','77','88','99','aa','bb','cc','dd','ee','ff']
            plt.plot(ep_pos_x[idx],ep_pos_y[idx])
            for ddd in range(len(ep_pos_x[idx])):
                if ep_pos_xy[idx].count([ep_pos_x[idx][ddd],ep_pos_y[idx][ddd]]) > 15:
                    G ='00'
                else:
                    G = color_code[-ep_pos_xy[idx].count([ep_pos_x[idx][ddd],ep_pos_y[idx][ddd]])]
                #print("#"+str(R)+str(G)+str(B))
                #ax.axvspan(ep_pos_x[idx][ddd]+0.25,ep_pos_x[idx][ddd]-0.25,(ep_pos_y[idx][ddd]/10)+0.5-0.025,(ep_pos_y[idx][ddd]/10)+0.5+0.025, color = "#"+R+G+B)
                ax.axvspan(ep_pos_x[idx][ddd]+0.25,ep_pos_x[idx][ddd]-0.25,(ep_pos_y[idx][ddd]/10)+1.0-0.025,(ep_pos_y[idx][ddd]/10)+1.0+0.025, color = "#"+R+G+B)
            c = patches.Circle(xy=(ep_pos_x[idx][0], ep_pos_y[idx][0]), radius=0.25, fc='b', ec='r')
            ax.add_patch(c)
            c = patches.Circle(xy=(ep_target[idx][0], ep_target[idx][1]), radius=0.25, fc='g', ec='r')
            ax.add_patch(c)
            ax.set_aspect('equal',adjustable='box')
            #plt.grid(which='major',color='black',linestyle='-')
            plt.show()
            fig.savefig(directory+'/' + str(name_video[idx_name])+".png")
            print("Created")
            #num+=1 

            create_marged_view(scene_scope, name_video[idx_name], directory, ep_start[idx],task_scope)
    
    def extract_input(self, env, device):
            state = {
                "current": env.render('resnet_features'),
                "goal": env.render_target('word_features')
            }
            # if self.method == 'word2vec_notarget' or self.method=='Transformer_word2vec_notarget' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec"or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat"or self.method=="Transformer_word2vec_notarget_action" or self.method=="Transformer_word2vec_notarget_word2vec_action"or self.method=="grid_memory"or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=="grid_memory_action": #origin
            if self.method == 'word2vec_notarget' or self.method=='Transformer_word2vec_notarget' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec"or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat"or self.method=="Transformer_word2vec_notarget_action" or self.method=="Transformer_word2vec_notarget_word2vec_action"or self.method=="grid_memory"or self.method=="grid_memory_no_observation"or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=="grid_memory_action"or self.method=="scene_only": #add
                state["object_mask"] = env.render_mask_similarity()
                x_processed = torch.from_numpy(state["current"])
                object_mask = torch.from_numpy(state['object_mask'])

                x_processed = x_processed.to(device)
                object_mask = object_mask.to(device)

                return state, x_processed, object_mask

    def run(self):
        scene_stats = dict()
        scene_stats2 = dict()
        scene_success = dict()
        resultData = []
        resultData2 = []
        #for scene_scope, items in TASK_LIST.items():#scene_scope are bathroom or bedroom or etc.. item is number suhch as 27
        for scene_scope, items in self.config['task_list'].items():
            if len(self.config['test_scenes']) != 0 and not scene_scope in self.config['test_scenes']:
                continue

            if self.SSL:
                scene_net = self.scene_nets[scene_scope]
            else:
                scene_net = self.scene_nets
            # if self.method=='word2vec_notarget' or self.method=='Baseline' or self.method=='Transformer_word2vec_notarget' or self.method =='Transformer_Concat' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec"or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat"or self.method=="Transformer_word2vec_notarget_action" or self.method=="Transformer_word2vec_notarget_word2vec_action"or self.method=="grid_memory"or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=="grid_memory_action": #origin
            if self.method=='word2vec_notarget' or self.method=='Baseline' or self.method=='Transformer_word2vec_notarget' or self.method =='Transformer_Concat' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec"or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat"or self.method=="Transformer_word2vec_notarget_action" or self.method=="Transformer_word2vec_notarget_word2vec_action"or self.method=="grid_memory"or self.method=="grid_memory_no_observation"or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=="grid_memory_action"or self.method=="scene_only": #add
                scene_stats[scene_scope] = dict()
                scene_stats[scene_scope]["length"] = list()
                scene_stats[scene_scope]["spl"] = list()
                scene_stats[scene_scope]["success"] = list()
                scene_stats[scene_scope]["spl_long"] = list()
                scene_stats[scene_scope]["success_long"] = list()
                scene_stats[scene_scope]["failure_lost"] = list()
                scene_stats[scene_scope]["failure_done_visible"] = list()
            scene_stats2[scene_scope] = list()
            scene_success[scene_scope] = list()
            for task_scope in items:
                try: #add
                    # if self.method=='word2vec_notarget' or self.method=='Transformer_word2vec_notarget' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec"or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat"or self.method=="Transformer_word2vec_notarget_action" or self.method=="Transformer_word2vec_notarget_word2vec_action"or self.method=="grid_memory"or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=="grid_memory_action": #origin
                    if self.method=='word2vec_notarget' or self.method=='Transformer_word2vec_notarget' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec"or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat"or self.method=="Transformer_word2vec_notarget_action" or self.method=="Transformer_word2vec_notarget_word2vec_action"or self.method=="grid_memory"or self.method=="grid_memory_no_observation"or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=="grid_memory_action"or self.method=="scene_only": #add
                        env = THORDiscreteEnvironment(
                                                    scene_name=scene_scope,
                                                    terminal_state=task_scope,
                                                    h5_file_path=(lambda scene: self.config.get("h5_file_path", "/app/data/{scene}.h5").replace('{scene}', scene)),
                                                    method=self.method
                                                    )
                    elif self.method =='Baseline'or self.method =='Transformer_Concat':
                        env = THORDiscreteEnvironment(scene_name = scene_scope,
                                                    terminal_state_id = int(task_scope),
                                                    h5_file_path=(lambda scene: self.config.get("h5_file_path", "/app/data/{scene}.h5").replace('{scene}', scene)),
                                                    method=self.method
                                                    )
                    else:
                        print('error')
                    #env = THORDiscreteEnvironment(
                    #    scene_name=scene_scope,
                    #    h5_file_path=(lambda scene: self.config.get("h5_file_path", "/app/data/{scene}.h5").replace('{scene}', scene)),
                    #    terminal_state_id=int(task_scope),
                    #)

                    graph = env._get_graph_handle()
                    #hitting_times = graph['hitting_times'][()]
                    shortest_paths = graph['shortest_path_distance'][()]

                    ep_rewards = []
                    ep_lengths = []
                    ep_collisions = []
                    ep_normalized_lengths = []
                    ep_actions = []#Rui
                    ep_start = []#Rui
                    ep_success = [] # Rui
                    ep_pos_x = list()#Rui
                    ep_pos_y = list()#Rui
                    ep_pos_xy = list()#rui
                    ep_target = [] #Rui
                    ep_spl = []
                    ep_shortest_distance = []
                    ep_enco = []
                    ep_deco = []
                    ep_fail_threshold = 300
                    # if self.method=='word2vec_notarget' or self.method=='Baseline' or self.method=='Transformer_word2vec_notarget'or self.method =='Transformer_Concat' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec"or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat"or self.method=="Transformer_word2vec_notarget_action" or self.method=="Transformer_word2vec_notarget_word2vec_action"or self.method=="grid_memory"or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=="grid_memory_action": #origin
                    if self.method=='word2vec_notarget' or self.method=='Baseline' or self.method=='Transformer_word2vec_notarget'or self.method =='Transformer_Concat' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec"or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat"or self.method=="Transformer_word2vec_notarget_action" or self.method=="Transformer_word2vec_notarget_word2vec_action"or self.method=="grid_memory"or self.method=="grid_memory_no_observation"or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=="grid_memory_action"or self.method=="scene_only": #add
                        ep_shortest_distance = []
                        embedding_vectors = []
                        state_ids = list()
                    # for (i_episode, start) in enumerate(env.get_initial_states(int(task_scope))):
                    #for (i_episode, start) in enumerate(random.sample(env.get_initial_states(int(task_scope)),NUM_EVAL_EPISODES)):
                    random.seed(0) #add(20220526) fix start 
                    for (i_episode, start) in enumerate(random.sample(env.get_initial_states(self.method,task_scope),self.config['num_episode'])): #main
                        # print("i_episode : {}, start : {}".format(i_episode, start)) #add check
                    #for (i_episode, start) in enumerate(random.sample(env.get_initial_states(self.method,int(task_scope)),self.config['num_episode'])):
                    #for i_episode in range(self.config['num_episode']):
                        #if not env.reset():
                        #    continue
                        #if start%12<=3:
                        #    env.reset(initial_state_id = start)
                        #else:
                        #    continue
                        #print(start)
                        env.reset(initial_state_id =start)
                        terminal = False
                        ep_reward = 0
                        ep_collision = 0
                        ep_t = 0
                        actions = [] #Rui
                        pos_x = list()#Rui
                        pos_y = list()#rui
                        pos_xy = list()
                        ep_start.append(env.current_state_id)
                        #hitting_time = hitting_times[start, int(task_scope)]
                        #shortest_path = shortest_paths[start, int(task_scope)]
                        #shortest_path = shortest_paths[start, task_scope]
                        #print(i_episode)
                        #print(env.locations[env.current_state_id])
                        #ini_pos_x = env.locations[env.current_state_id][0]
                        #ini_pos_y = env.locations[env.current_state_id][2] # IF ITS ORIGONAL DATA y=[1]
                        left_pos_x = np.min(env.locations,axis=0)[0]-1.5
                        left_pos_y = np.max(env.locations,axis=0)[2]+1.5# IF ITS ORIGONAL DATA y=[1]
                        ini_pos_x = env.locations[env.current_state_id][0]-left_pos_x
                        ini_pos_y = env.locations[env.current_state_id][2]-left_pos_y # IF ITS ORIGONAL DATA y=[1]
                        pos_terminal_x = env.locations[env.terminal_state_id][0]-left_pos_x
                        pos_terminal_y = env.locations[env.terminal_state_id][2]-left_pos_y
                        ep_target.append([pos_terminal_x,pos_terminal_y])

                        pos_x.append(ini_pos_x)
                        pos_y.append(ini_pos_y)
                        enco_attens = []
                        deco_attens = []
                        if self.add_reward == 'count':
                            self.states = [[-1] for i in range(self.memory_size)]
                            self.actions = [[-1] for i in range(self.memory_size)]
                            self.prev_states = [[-1] for i in range(self.memory_size)]
                        # if self.method =="Transformer_Concat" or self.method =="Transformer_Sum" or self.method=='Transformer_word2vec_notarget' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec"or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat"or self.method=="Transformer_word2vec_notarget_action" or self.method=="Transformer_word2vec_notarget_word2vec_action"or self.method=="grid_memory"or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=="grid_memory_action": #origin
                        if self.method =="Transformer_Concat" or self.method =="Transformer_Sum" or self.method=='Transformer_word2vec_notarget' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec"or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat"or self.method=="Transformer_word2vec_notarget_action" or self.method=="Transformer_word2vec_notarget_word2vec_action"or self.method=="grid_memory"or self.method=="grid_memory_no_observation"or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=="grid_memory_action"or self.method=="scene_only":
                            memory = torch.zeros(self.memory_size, self.embed_size)            
                            mask = torch.ones(self.memory_size)
                            theta = 0
                            positions = [[0,0, theta] for i in range(self.memory_size)]
                        # if self.method=="grid_memory"or self.method=="grid_memory_action": #origin
                        if self.method=="grid_memory"or self.method=="grid_memory_no_observation"or self.method=="grid_memory_action"or self.method=="scene_only": #add
                            grid_memory = torch.zeros(self.memory_size, 16, 16)
                            grid_mask = torch.ones(self.memory_size)
                        if self.method=="Transformer_word2vec_notarget_action" or self.method=="Transformer_word2vec_notarget_word2vec_action"or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=="grid_memory_action":
                            #act_memory = torch.zeros(self.memory_size, 8)
                            #act_mask = torch.ones(self.memory_size)
                            #last_act = torch.zeros(8).to(self.device)
                            act_memory = torch.zeros(self.memory_size, 128)
                            act_mask = torch.ones(self.memory_size)
                            last_act = torch.zeros(128).to(self.device)
                        if self.method=='Transformer_word2vec_notarget_word2vec_action_posi':
                            locs = np.zeros((self.memory_size, 8))
                            rllocs = np.zeros((self.memory_size, 4))
                            deg = np.zeros((1, 1))
                            nlocs = 0
                        

                        while not terminal:
                            if self.method=='Transformer_word2vec_notarget_action_posi':
                                home_posi = np.array([env.location]) / 5
                                #print(self.deg)
                                home_posi = np.append(home_posi, self.deg, axis=1)
                                home_posi = np.append(home_posi, self.deg, axis=1)
                                nlocs += 1
                                #print(home_posi)
                                #print(self.locs.shape)
                                locs = np.delete(locs, -1, axis=0)
                                #print(self.locs.shape)
                                locs = np.append(home_posi, locs, axis=0)
                                #print(self.locs)
                                if nlocs < self.memory_size:
                                    rllocs[0:nlocs] = locs[0:nlocs] - home_posi
                                    #print(self.rllocs)
                                    for m in range(nlocs):
                                        rllocs[m][2] = np.sin(np.deg2rad(rllocs[m][2]))
                                        rllocs[m][3] = np.cos(np.deg2rad(rllocs[m][3]))
                                        #print(self.rllocs)
                                else:
                                    self.rllocs[:] = self.locs[:] - home_posi
                                    #print(self.rllocs)
                                    for m in range(self.memory_size):
                                        rllocs[m][2] = np.sin(np.deg2rad(rllocs[m][2]))
                                        rllocs[m][3] = np.cos(np.deg2rad(rllocs[m][3]))
                                        #print(self.rllocs)
                                        #print(self.locs)
                                        #print(self.rllocs)
                            if self.method == "Baseline" :
                                state = torch.Tensor(env.render(mode='resnet_features'))
                                target = torch.Tensor(env.render_target(mode='resnet_features'))
                                state = state.to(self.device)                        
                                target = target.to(self.device)                        
                                (policy, value,) = scene_net.forward(self.shared_net.forward((state, target,)))
                            elif self.method =='word2vec_notarget' or self.method=='Transformer_word2vec_notarget' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec"or self.method=="Transformer_word2vec_notarget_word2vec_concat":
                                state, x_processed, object_mask = self.extract_input(env, self.device)
                                if self.Posi:
                                    (policy, value, memory, mask) = scene_net.forward(self.shared_net.forward((x_processed[:,-1], object_mask, memory, mask, self.device, positions)))
                                elif self.Key=="word2vec":
                                    (policy, value, memory, mask, encoder_atten_weights, decoder_atten_weights) = scene_net.forward(self.shared_net.forward((x_processed[:,-1], object_mask, memory, mask, self.device, env.s_target)))
                                else:
                                    (policy, value, memory, mask,encoder_atten_weights,decoder_atten_weights) = scene_net.forward(self.shared_net.forward((x_processed[:,-1], object_mask, memory, mask, self.device)))
                                if self.device != "cpu":
                                    memory = memory.clone().detach().cpu()
                                    memory = memory.view(self.memory_size, self.embed_size)
                                    mask = mask.clone().detach().cpu()
                                    mask = mask.view(self.memory_size)
                                else:
                                    memory = memory.view(self.memory_size, self.embed_size)
                                    mask = mask.view(self.memory_size)
                            elif self.method=="Transformer_word2vec_notarget_word2vec_posi":
                                #print(self.device)
                                state, x_processed, object_mask = self.extract_input(env, self.device)
                                (policy, value, memory, mask, encoder_atten_weights, decoder_atten_weights) = scene_net.forward(self.shared_net.forward((x_processed[:,-1], object_mask, memory, mask, self.device, positions, env.s_target)))
                                if self.device != "cpu":
                                    memory = memory.clone().detach().cpu()
                                    memory = memory.view(self.memory_size, self.embed_size)
                                    mask = mask.clone().detach().cpu()
                                    mask = mask.view(self.memory_size)
                                else:
                                    memory = memory.view(self.memory_size, self.embed_size)
                                    mask = mask.view(self.memory_size)
                            elif self.method=="Transformer_word2vec_notarget_action":
                                state, x_processed, object_mask = self.extract_input(env, self.device)
                                (policy, value, memory, mask, act_memory, act_mask, encoder_atten_weights, decoder_atten_weights) = scene_net.forward(self.shared_net.forward((x_processed[:,-1], object_mask, memory, mask, self.device, last_act, act_memory, act_mask)))
                                if self.device != "cpu":
                                    memory = memory.clone().detach().cpu()
                                    memory = memory.view(self.memory_size, self.embed_size)
                                    mask = mask.clone().detach().cpu()
                                    mask = mask.view(self.memory_size)
                                    act_memory = act_memory.clone().detach().cpu()
                                    #act_memory = act_memory.view(self.memory_size, 8)
                                    act_memory = act_memory.view(self.memory_size, 128)
                                    act_mask = act_mask.clone().detach().cpu()
                                    act_mask = act_mask.view(self.memory_size)
                                else:
                                    memory = memory.view(self.memory_size, self.embed_size)
                                    mask = mask.view(self.memory_size)
                                    #act_memory = act_memory.view(self.memory_size, 8)
                                    act_memory = act_memory.view(self.memory_size, 128)
                                    act_mask = act_mask.view(self.memory_size)
                            elif self.method=="Transformer_word2vec_notarget_word2vec_action":
                                state, x_processed, object_mask = self.extract_input(env, self.device)
                                (policy, value, memory, mask, act_memory, act_mask, encoder_atten_weights, decoder_atten_weights) = scene_net.forward(self.shared_net.forward((x_processed[:,-1], object_mask, memory, mask, self.device, env.s_target, last_act, act_memory, act_mask)))
                                if self.device != "cpu":
                                    memory = memory.clone().detach().cpu()
                                    memory = memory.view(self.memory_size, self.embed_size)
                                    mask = mask.clone().detach().cpu()
                                    mask = mask.view(self.memory_size)
                                    act_memory = act_memory.clone().detach().cpu()
                                    act_memory = act_memory.view(self.memory_size, 8)
                                    act_mask = act_mask.clone().detach().cpu()
                                    act_mask = act_mask.view(self.memory_size)
                                else:
                                    memory = memory.view(self.memory_size, self.embed_size)
                                    mask = mask.view(self.memory_size)
                                    act_memory = act_memory.view(self.memory_size, 8)
                                    act_mask = act_mask.view(self.memory_size)
                            elif self.method=="Transformer_word2vec_notarget_word2vec_action_posi":
                                state, x_processed, object_mask = self.extract_input(env, self.device)
                                (policy, value, memory, mask, act_memory, act_mask, encoder_atten_weights, decoder_atten_weights) = scene_net.forward(self.shared_net.forward((x_processed[:,-1], object_mask, memory, mask, self.device, env.s_target, last_act, act_memory, act_mask, rllocs)))
                                if self.device != "cpu":
                                    memory = memory.clone().detach().cpu()
                                    memory = memory.view(self.memory_size, self.embed_size)
                                    mask = mask.clone().detach().cpu()
                                    mask = mask.view(self.memory_size)
                                    act_memory = act_memory.clone().detach().cpu()
                                    act_memory = act_memory.view(self.memory_size, 8)
                                    act_mask = act_mask.clone().detach().cpu()
                                    act_mask = act_mask.view(self.memory_size)
                                else:
                                    memory = memory.view(self.memory_size, self.embed_size)
                                    mask = mask.view(self.memory_size)
                                    act_memory = act_memory.view(self.memory_size, 8)
                                    act_mask = act_mask.view(self.memory_size)
                            # elif self.method=="grid_memory": #origin
                            elif self.method=="grid_memory" or self.method=="grid_memory_no_observation"or self.method=="scene_only":
                                state, x_processed, object_mask = self.extract_input(env, self.device)
                                (policy, value, memory, mask, grid_memory, grid_mask, encoder_atten_weights, decoder_atten_weights) = scene_net.forward(self.shared_net.forward((x_processed[:,-1], object_mask, memory, mask, env.s_target, grid_memory, grid_mask, self.device)))
                                if self.device != "cpu":
                                    memory = memory.clone().detach().cpu()
                                    memory = memory.view(self.memory_size, self.embed_size)
                                    mask = mask.clone().detach().cpu()
                                    mask = mask.view(self.memory_size)
                                    grid_memory = grid_memory.clone().detach().cpu()
                                    grid_memory = grid_memory.view(self.memory_size, 16, 16)
                                    grid_mask = grid_mask.clone().detach().cpu()
                                    grid_mask = grid_mask.view(self.memory_size)
                                else:
                                    memory = memory.view(self.memory_size, self.embed_size)
                                    mask = mask.view(self.memory_size)
                                    grid_memory = grid_memory.view(self.memory_size, 16, 16)
                                    grid_mask = grid_mask.view(self.memory_size)
                            elif self.method=="grid_memory_action":
                                state, x_processed, object_mask = self.extract_input(env, self.device)
                                (policy, value, memory, mask, grid_memory, grid_mask, act_memory, act_mask, encoder_atten_weights, decoder_atten_weights) = scene_net.forward(self.shared_net.forward((x_processed[:,-1], object_mask, memory, mask, env.s_target, grid_memory, grid_mask, last_act, act_memory, act_mask, self.device)))
                                if self.device != "cpu":
                                    memory = memory.clone().detach().cpu()
                                    memory = memory.view(self.memory_size, self.embed_size)
                                    mask = mask.clone().detach().cpu()
                                    mask = mask.view(self.memory_size)
                                    grid_memory = grid_memory.clone().detach().cpu()
                                    grid_memory = grid_memory.view(self.memory_size, 16, 16)
                                    grid_mask = grid_mask.clone().detach().cpu()
                                    grid_mask = grid_mask.view(self.memory_size)
                                    act_memory = act_memory.clone().detach().cpu()
                                    act_memory = act_memory.view(self.memory_size, 128)
                                    act_mask = act_mask.clone().detach().cpu()
                                    act_mask = act_mask.view(self.memory_size)
                                else:
                                    memory = memory.view(self.memory_size, self.embed_size)
                                    mask = mask.view(self.memory_size)
                                    grid_memory = grid_memory.view(self.memory_size, 16, 16)
                                    grid_mask = grid_mask.view(self.memory_size)
                                    act_memory = act_memory.view(self.memory_size, 32)
                                    act_mask = act_mask.view(self.memory_size)
                            else:
                                state = torch.Tensor(env.render(mode='resnet_features'))
                                target = torch.Tensor(env.render_target(mode='resnet_features'))
                                state = state.to(self.device)                        
                                target = target.to(self.device)                        
                                (policy, value, memory, mask) = scene_net.forward(self.shared_net.forward((state[:,-1], target[:,-1],memory, mask, self.device)))
                                memory = memory.clone().detach().cpu()
                                memory = memory.view(self.memory_size, self.embed_size)
                                mask = mask.clone().detach().cpu()
                                mask = mask.view(self.memory_size)
                            encoder_atten_weights = encoder_atten_weights.clone().detach().cpu().numpy()
                            decoder_atten_weights = decoder_atten_weights.clone().detach().cpu().numpy()
                            with torch.no_grad():
                                action = F.softmax(policy, dim=0).multinomial(1).data.cpu().numpy()[0]

                            #if self.add_reward == 'count' and action!= 8:
                            #    del self.prev_states[0]
                            #    self.prev_states.append(env.current_state_id)
                            if self.add_reward == 'count' and action!= 8:
                                del self.actions[0]
                                self.actions.append(action)
                                LIST = [i for i, x in enumerate(self.prev_states) if x == env.current_state_id]
                                for e in LIST:
                                    if self.actions[e] == action:
                                        #reward += -0.05
                                        ep_reward += -0.1

                            env.step(action)
                            actions.append(action)#Rui
                            if self.Posi:
                                del positions[0]
                                if env.collided:
                                    positions.append([0,0, theta])
                                elif action == 0:
                                    positions.append([1,0, theta])
                                elif action == 3:
                                    positions.append([-1,0,theta])
                                elif action == 6:
                                    positions.append([0,1,theta])
                                elif action == 7:
                                    positions.append([0,-1,theta])
                                elif action == 1:
                                    theta+=1
                                    positions.append([0,0,theta])
                                elif action == 2:
                                    theta-=1
                                    positions.append([0,0,theta])
                                else:
                                    positions.append([0,0,theta])
                            if self.method =="Transformer_word2vec_notarget_action" or self.method =="Transformer_word2vec_notarget_word2vec_action" or self.method =="Transformer_word2vec_notarget_word2vec_action_posi" or self.method =='grid_memory_action':
                                if action == 0:
                                    last_act= torch.tensor(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                elif action == 1:
                                    last_act = torch.tensor(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                elif action == 2:
                                    last_act = torch.tensor(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                elif action == 3:
                                    last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                elif action == 4:
                                    last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                elif action == 5:
                                    last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                elif action == 6:
                                    last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), dtype=torch.float32).to(self.device)
                                elif action == 7:
                                    last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), dtype=torch.float32).to(self.device)
                                #if action == 0:
                                #    last_act == torch.tensor(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                #elif action == 1:
                                #    last_act == torch.tensor(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                #elif action == 2:
                                #    last_act == torch.tensor(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                #elif action == 3:
                                #    last_act == torch.tensor(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                #elif action == 4:
                                #    last_act == torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                #elif action == 5:
                                #    last_act == torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                #elif action == 6:
                                #    last_act == torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), dtype=torch.float32).to(self.device)
                                #elif action == 7:
                                #    last_act == torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), dtype=torch.float32).to(self.device)
                                #if action == 0:
                                #    last_act = torch.tensor(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                #elif action == 1:
                                #    last_act = torch.tensor(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                #elif action == 2:
                                #    last_act = torch.tensor(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                #elif action == 3:
                                #    last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                #elif action == 4:
                                #    last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                #elif action == 5:
                                #    last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), dtype=torch.float32).to(self.device)
                                #elif action == 6:
                                #    last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), dtype=torch.float32).to(self.device)
                                #elif action == 7:
                                #    last_act = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), dtype=torch.float32).to(self.device)

                            if self.method =="Transformer_word2vec_notarget_word2vec_action_posi":
                                if action==1:
                                    deg+=45.0
                                elif action==2:
                                    deg -=45.0
                            if env.collided: ep_collision += 1
                            ep_reward += env.reward
                            #if self.add_reward == 'count' and action!= 8:
                            #    del self.states[0]
                            #    self.states.append(env.current_state_id)
                            #    n = self.states.count(env.current_state_id)
                            #    ep_reward += -(1.5**(n-5)/100.0) 
                            
                            if self.add_reward == 'count' and action!= 8:
                                del self.actions[0]
                                self.actions.append(action)
                                #LIST = [i for i, x in enumerate(self.prev_states) if x == env.current_state_id]
                                #for e in LIST:
                                #    if self.actions[e] == action:
                                #        #ep_reward += -0.1
                                #        #ep_reward += -0.05
                                #        ep_reward += -0.03
                                #LIST = [i for i, x in enumerate(self.prev_states) if x == self.env.current_state_id]
                                LIST = [i for i, x in enumerate(self.prev_states) if x == self.prev_states[-1]]
                                CCC = 0 
                                for e in LIST:
                                    if self.actions[e] == action:
                                        CCC +=1
                                if CCC >= 2:
                                    #reward += -0.01
                                    ep_reward += -(0.01 * (CCC-1))
                                    #reward += -0.05
                                    #reward += -0.1
                                    #pass
                                        
                            #if self.add_reward == 'count' and action!= 8:
                            #    self.states.append(env.current_state_id)
                            #    n = self.states.count(env.current_state_id)
                            #    ep_reward += 0.01 -(math.sqrt(n)/100.0) 
                            #ep_reward += env.reward
                            terminal = env.is_terminal
                            ep_t += 1 
                            Pos_x = env.locations[env.current_state_id][0]-left_pos_x
                            Pos_y = env.locations[env.current_state_id][2]-left_pos_y
                            pos_x.append(Pos_x)
                            pos_y.append(Pos_y)
                            Pos_xy = [Pos_x,Pos_y]
                            pos_xy.append(Pos_xy)
                            enco_attens.append(encoder_atten_weights)
                            deco_attens.append(decoder_atten_weights)
                            # if self.method =='word2vec_notarget' or self.method=='Transformer_word2vec_notarget' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec" or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat" or self.method=="Transformer_word2vec_notarget_action"or self.method =="Transformer_word2vec_notarget_word2vec_action"or self.method =="grid_memory"or self.method =="Transformer_word2vec_notarget_word2vec_action_posi" or self.method=="grid_memory_action": #origin
                            if self.method =='word2vec_notarget' or self.method=='Transformer_word2vec_notarget' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec" or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat" or self.method=="Transformer_word2vec_notarget_action"or self.method =="Transformer_word2vec_notarget_word2vec_action"or self.method =="grid_memory"or self.method =="grid_memory_no_observation"or self.method =="Transformer_word2vec_notarget_word2vec_action_posi" or self.method=="grid_memory_action"or self.method =="scene_only": #add
                                if ep_t >= ep_fail_threshold:##Rui
                                    terminal = True#Rui
                            else:
                                if ep_t >= 250:
                                    terminal = True


                        # if self.method =='word2vec_notarget' or self.method=='Transformer_word2vec_notarget' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec" or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat"or self.method=="Transformer_word2vec_notarget_action"or self.method =="Transformer_word2vec_notarget_word2vec_action"or self.method =="grid_memory"or self.method =="Transformer_word2vec_notarget_word2vec_action_posi"or self.method=="grid_memory_action": #origin
                        if self.method =='word2vec_notarget' or self.method=='Transformer_word2vec_notarget' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec" or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat"or self.method=="Transformer_word2vec_notarget_action"or self.method =="Transformer_word2vec_notarget_word2vec_action"or self.method =="grid_memory"or self.method =="grid_memory_no_observation"or self.method =="Transformer_word2vec_notarget_word2vec_action_posi"or self.method=="grid_memory_action"or self.method =="scene_only": #add
                            if env.success:
                                ep_success.append(True)
                            else:
                                ep_success.append(False)
                        else:
                            if ep_t < 250:
                                ep_success.append(True)
                            else:
                                ep_success.append(False)
                        ep_lengths.append(ep_t)
                        ep_rewards.append(ep_reward)
                        ep_collisions.append(ep_collision)
                        ep_actions.append(actions)#Rui
                        ep_pos_x.append(pos_x)#rui
                        ep_pos_y.append(pos_y)#rui
                        ep_pos_xy.append(pos_xy)
                        ep_enco.append(enco_attens)
                        ep_deco.append(deco_attens)
                        ep_shortest_distance.append(env.shortest_path_terminal(ep_start[-1]))

                        # Compute SPL
                        spl = env.shortest_path_terminal(ep_start[-1]) / ep_t
                        #print(env.shortest_path_terminal(ep_start[-1]))
                        ep_spl.append(spl)
                        #if VERBOSE: print("episode #{} ends after {} steps".format(i_episode, ep_t)) #origin
                        if VERBOSE:
                            # with open("test.log", "a", encoding="utf-8") as f:
                            # print("######### {} : ".format(count_files)) # count ok?
                            with open("./tmp/output_log_path_{}.txt".format(count_files), mode="r", encoding="utf-8") as f:
                                p_new = f.readline()
                                # p_new = str('"' + str(p_new) + '"')
                                # print(f"read test : {p_new}")
                                p_new = pathlib.Path(p_new)

                            with p_new.open(mode="a", encoding="utf-8") as f:
                                # output_steps = ("episode #{} ends after {} steps".format(i_episode, ep_t)) #ADD origin
                                output_steps = ("episode #{} ends after {} steps, success : {}".format(i_episode, ep_t, ep_success[-1])) #ADD(20220525) 
                                f.write(f"{output_steps}\n")#ADD
                                print(output_steps)#ADD
                    #if VERBOSE:
                    #    with open("test.log", "a", encoding="utf-8") as f:
                    #        output_eval = ('evaluation: %s %s' % (scene_scope, task_scope))#ADD
                    #        f.write(f"{output_eval}\n")#ADD
                    #        print(output_eval)#ADD 
                    #print('evaluation: %s %s' % (scene_scope, task_scope))#origin
                    # if self.method =='word2vec_notarget' or self.method=='Transformer_word2vec_notarget' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec" or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat"or self.method=="Transformer_word2vec_notarget_action"or self.method =="Transformer_word2vec_notarget_word2vec_action"or self.method =="grid_memory"or self.method =="Transformer_word2vec_notarget_word2vec_action_posi"or self.method=="grid_memory_action": #origin
                    if self.method =='word2vec_notarget' or self.method=='Transformer_word2vec_notarget' or self.method=="Transformer_word2vec_notarget_withposi"or self.method=="Transformer_word2vec_notarget_word2vec" or self.method=="Transformer_word2vec_notarget_word2vec_posi"or self.method=="Transformer_word2vec_notarget_word2vec_concat"or self.method=="Transformer_word2vec_notarget_action"or self.method =="Transformer_word2vec_notarget_word2vec_action"or self.method =="grid_memory"or self.method =="grid_memory_no_observation"or self.method =="Transformer_word2vec_notarget_word2vec_action_posi"or self.method=="grid_memory_action"or self.method =="scene_only":
                        ind_succeed_ep = [
                            i for (i, ep_suc) in enumerate(ep_success) if ep_suc]
                        #print(ind_succeed_ep)
                        ep_rewards = np.array(ep_rewards)
                        ep_lengths = np.array(ep_lengths)
                        ep_collisions = np.array(ep_collisions)
                        ep_spl = np.array(ep_spl)
                        ep_start = np.array(ep_start)
                        ep_success_percent = (
                            (len(ind_succeed_ep) / self.config['num_episode']) * 100)
                        ep_spl_mean = np.sum(ep_spl[ind_succeed_ep]) / self.config['num_episode']
                        # Stat on long path
                        ind_succeed_far_start = []
                        ind_far_start = []
                        for i, short_dist in enumerate(ep_shortest_distance):
                            if short_dist > 5:
                                if ep_success[i]:
                                    ind_succeed_far_start.append(i)
                                ind_far_start.append(i)

                        nb_long_episode = len(ind_far_start)
                        if nb_long_episode == 0:
                            nb_long_episode = 1
                            ep_success_long_percent = (
                                (len(ind_succeed_far_start) / nb_long_episode) * 100)
                            # ep_spl_long_mean = 99999 #origin
                            ep_spl_long_mean = np.nan #add
                        else:
                            ep_success_long_percent = (
                                (len(ind_succeed_far_start) / nb_long_episode) * 100)
                            ep_spl_long_mean = np.sum(ep_spl[ind_succeed_far_start]) / nb_long_episode
                        scene_stats[scene_scope]["length"].extend(
                            ep_lengths[ind_succeed_ep])
                        scene_stats[scene_scope]["spl"].append(ep_spl_mean)
                        scene_stats[scene_scope]["success"].append(
                            ep_success_percent)
                        scene_stats[scene_scope]["spl_long"].append(
                            ep_spl_long_mean)
                        scene_stats[scene_scope]["success_long"].append(
                            ep_success_long_percent)

                        tmpData = [np.mean(
                            ep_rewards), np.mean(ep_lengths), np.mean(ep_collisions), ep_success_percent, ep_spl, ind_succeed_ep]
                        resultData = np.hstack((resultData, tmpData))
                    else:
                        ind_succeed_ep = [
                            i for (i, ep_suc) in enumerate(ep_success) if ep_suc]
                        ep_success_percent = (
                            (len(ind_succeed_ep) / len(ep_success)) * 100)
                        ep_spl = np.array(ep_spl)
                        ep_spl_mean = np.sum(ep_spl[ind_succeed_ep]) / NUM_EVAL_EPISODES

                    # Save failed episode
                    ind_failed_ep = [
                        i for (i, ep_suc) in enumerate(ep_success) if not ep_suc]
                    ep_rewards = np.array(ep_rewards)
                    ep_lengths = np.array(ep_lengths)
                    ep_collisions = np.array(ep_collisions)
                    ep_spl = np.array(ep_spl)
                    ep_start = np.array(ep_start)
                    #self.create_topdown(ep_lengths, ep_start, ep_pos_x,ep_pos_y,ep_pos_xy,scene_scope,task_scope,ep_target)
                    
                    # self.save_video(ep_lengths, ep_actions, ep_start, env, scene_scope, task_scope)##Rui
                    # if self.method =='word2vec_notarget' or self.method=='Transformer_word2vec_notarget':
                    #    self.save_video_word2vec(ep_lengths, ep_actions, ep_start, ind_succeed_ep, env, scene_scope, task_scope)
                    # # elif self.method =='Transformer_word2vec_notarget_word2vec_posi'or self.method =='Transformer_word2vec_notarget_word2vec_action' or self.method =='grid_memory' or self.method=="Transformer_word2vec_notarget_word2vec": #origin
                    # elif self.method =='Transformer_word2vec_notarget_word2vec_posi'or self.method =='Transformer_word2vec_notarget_word2vec_action' or self.method =='grid_memory' or self.method =='grid_memory_no_observation' or self.method=="Transformer_word2vec_notarget_word2vec": #add
                    #    self.save_video_word2vec(ep_lengths, ep_actions, ep_start, ind_succeed_ep, env, scene_scope, task_scope,ep_enco, ep_deco)
                    # else:
                    #    self.save_video(ep_lengths, ep_actions, ep_start, env, scene_scope, task_scope)##Rui
                    if VERBOSE:
                        # with open("test.log", "a", encoding="utf-8") as f:
                        with p_new.open(mode="a", encoding="utf-8") as f:
                            #output_ep_s_per=('success rate: %.2f' % ep_success_percent)#add
                            output_ep_s_per=('episode success: {:.2f}% ({} / {})'.format(ep_success_percent, len(ind_succeed_ep), self.config['num_episode']))#add
                            output_ev=('evaluation: %s %s' % (scene_scope, task_scope))#add
                            output_ep_r=('mean episode reward: %.2f' % np.mean(ep_rewards))#add
                            output_ep_l=('mean episode length: %.2f' % np.mean(ep_lengths))#add
                            output_ep_c=('mean episode collision: %.2f' % np.mean(ep_collisions)) #add
                            output_ep_spl_m=('episode SPL: %.3f' % ep_spl_mean)#add
                            output_ep_s_l_per=('episode > 5 success: {:.2f}%'.format(ep_success_long_percent))#add
                            output_ep_spl_l_m=('episode SPL > 5: %.3f' % ep_spl_long_mean)#add
                            output_ep_nb=('nb episode > 5: {:.0f}'.format(nb_long_episode))#add
                            output_ep_fail=('episode failure: {:.2f}% ({} / {})'.format(100-ep_success_percent, self.config['num_episode']-len(ind_succeed_ep), self.config['num_episode'])) #add 20220524
                            #f.write(f"{output_ep_s_per}\n")#ADD
                            f.write(f"{output_ev}\n")#ADD
                            f.write(f"{output_ep_r}\n")#ADD
                            f.write(f"{output_ep_l}\n")#ADD
                            f.write(f"{output_ep_c}\n")#ADD
                            f.write(f"{output_ep_s_per}\n")#ADD
                            f.write(f"{output_ep_spl_m}\n")#ADD
                            f.write(f"{output_ep_s_l_per}\n")#ADD
                            f.write(f"{output_ep_spl_l_m}\n")#ADD
                            f.write(f"{output_ep_nb}\n")#ADD
                            f.write(f"{output_ep_fail}\n")#ADD 20220524
                            ### ADD(20220525) ###
                            ep_fail = len(ind_failed_ep)
                            
                            if ep_fail == 0:
                                ep_fail_lost = np.nan
                            else:
                                # Count number of fail with 300 step
                                ep_fail_lost = (np.count_nonzero(ep_lengths[ind_failed_ep] == ep_fail_threshold)/ep_fail)*100.0
                                f.write('episode failure lost %d%%\n' % (ep_fail_lost))
                                f.write('episode failure done %d%%\n' % (100-ep_fail_lost))
                                                
                            scene_stats[scene_scope]["failure_lost"].append(ep_fail_lost)

                            ind_done = []
                            for ind, e in enumerate(ep_lengths):
                                if ind in ind_failed_ep and e != ep_fail_threshold:
                                    ind_done.append(ind)
                            
                            ep_done_visible = 0
                            for i in ind_done:
                                env.reset()
                                # Set start position
                                env.current_state_id = ep_start[i]
                                for a in ep_actions[i]:
                                    env.step(a)
                                objects = [k.split("|")[0] for k in env.boudingbox.keys()]
                                if task_scope['object'] in objects:
                                    ep_done_visible += 1
                            if ind_done:
                                ep_done_visible = (ep_done_visible / len(ind_done))*100.0                    
                            f.write('episode failure done visible %d%%\n' % (ep_done_visible))
                            f.write("\n")#ADD 20220524
                            scene_stats[scene_scope]['failure_done_visible'].append(ep_done_visible)
                            ### ADD(20220525) ###
                    #print('success rate: %.2f' % (ep_success_percent))#origin
                    print('evaluation: %s %s' % (scene_scope, task_scope))#origin #important
                    print('mean episode reward: %.2f' % np.mean(ep_rewards))#origin
                    print('mean episode length: %.2f' % np.mean(ep_lengths))#origin
                    print('mean episode collision: %.2f' % np.mean(ep_collisions))#origin
                    print('episode success: {:.2f}% ({} / {})'.format(ep_success_percent, len(ind_succeed_ep), self.config['num_episode']))#origin
                    print('episode SPL: %.3f' % ep_spl_mean)
                    print('episode > 5 success: {:.2f}%'.format(ep_success_long_percent))#origin
                    print('episode SPL > 5: %.3f' % ep_spl_long_mean)#origin
                    print('nb episode > 5: {:.0f}'.format(nb_long_episode))#origin
                    print('episode failure:  {:.2f}% ({} / {})'.format(100-ep_success_percent, self.config['num_episode']-len(ind_succeed_ep), self.config['num_episode']))#add 20220524
                    print('episode failure lost %d%%' % (ep_fail_lost))#add 20220525
                    print('episode failure done %d%%' % (100-ep_fail_lost))#add 20220525    
                    print('episode failure done visible %d%%\n' % (ep_done_visible)) #add 20220525
                    scene_stats2[scene_scope].extend(ep_lengths)
                    scene_success[scene_scope].append(ep_success_percent)
                    #resultData2.append((scene_scope, str(task_scope), ep_success_percent, np.mean(ep_rewards), np.mean(ep_lengths), np.mean(ep_collisions),ep_spl_mean,))
                    resultData2.append((scene_scope, str(task_scope), ep_success_percent, np.mean(ep_rewards), np.mean(ep_lengths), np.mean(ep_collisions),ep_spl_mean,ep_success_long_percent, ep_spl_long_mean,))
                
                except:
                    error_point = ('!!! Error occured evaluation: %s %s !!!\n' % (scene_scope, task_scope))
                    with open(".error_point", mode="a", encoding="utf-8") as ff_error:
                        ff_error.write(error_point)
                    
                    continue

        print('\nResults (average trajectory length):')
        with open(self.config['csv_file'], 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Results (average trajectory length)"])
        for scene_scope in scene_stats2:
            print('%s: %.2f steps'%(scene_scope, np.mean(scene_stats2[scene_scope])))
            with open(self.config['csv_file'], 'a') as f:
                writer = csv.writer(f)
                writer.writerow([scene_scope,np.mean(scene_stats2[scene_scope]),"steps"])

        print('\nResults (average success rate):')
        with open(self.config['csv_file'], 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["Results (average success rate)"])
        for scene_scope in scene_success:
            print('%s: %.2f percent'%(scene_scope, np.mean(scene_success[scene_scope])))
            with open(self.config['csv_file'], 'a') as f:
                writer = csv.writer(f)
                writer.writerow([scene_scope,np.mean(scene_success[scene_scope]),"percent"])

        with open(self.config['csv_file'], 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["Scene", "Task", "Success percent", "Reward", "length", "Collision","SPL"])
        #if 'csv_file' in self.config and self.config['csv_file'] is not None:
        export_to_csv(resultData2, self.config['csv_file'])
