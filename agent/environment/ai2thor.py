# -*- coding: utf-8 -*-
import sys
import h5py
import json
import numpy as np
import random
import skimage.io
from skimage.transform import resize
from agent.environment.environment import Environment
from torchvision import transforms
from scipy import spatial

with open('.env', mode='r', encoding='utf-8') as f:
    target_path = "EXPERIMENT/" + f.readline().replace('\n', '')

json_open = open(target_path + "/"+ "param.json", "r")
json_load = json.load(json_open)

bbox_method = json_load['bbox_method']
print("bbox_method : {}".format(str(bbox_method))) #test

class THORDiscreteEnvironment(Environment):
    acts = ["MoveForward", "RotateRight", "RotateLeft", "MoveBackward", "LookUp", "LookDown", "MoveRight", "MoveLeft", "Done"]
    @staticmethod
    def _get_h5_file_path(h5_file_path, scene_name):
        if h5_file_path is None:
            h5_file_path = f"/app/data/{scene_name}.h5"
        elif callable(h5_file_path):
            h5_file_path = h5_file_path(scene_name)
        return h5_file_path

    def __init__(self, 
            method: str,
            scene_name = 'FloorPlan1',
            n_feat_per_location = 1,
            history_length : int = 4,
            screen_width = 224,
            screen_height = 224,
            terminal_state_id = 0,
            terminal_state=0,
            # bbox_method=None,#origin
            bbox_method=bbox_method,
            initial_state_id = None,
            h5_file_path = None,
            mask_size: int = 16,
            action_size:  int = 4,
            **kwargs):
        super(THORDiscreteEnvironment, self).__init__()
    
    #def __init__(self, 
    #        scene_name = 'FloorPlan1',
    #        n_feat_per_location = 1,
    #        history_length : int = 4,
    #        screen_width = 224,
    #        screen_height = 224,
    #        terminal_state_id = 0,
    #        initial_state_id = None,
    #        h5_file_path = None,
    #        **kwargs):
    #    super(THORDiscreteEnvironment, self).__init__()
    
        h5_file_path = THORDiscreteEnvironment._get_h5_file_path(h5_file_path, scene_name)
        self.terminal_state_id = terminal_state_id
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.n_feat_per_location = n_feat_per_location
        self.locations = self.h5_file['location'][()]
        self.rotations = self.h5_file['rotation'][()]
        self.history_length = history_length
        self.n_locations = self.locations.shape[0]
        self.terminals = np.zeros(self.n_locations)
        self.terminals[terminal_state_id] = 1
        self.terminal_states, = np.where(self.terminals)
        self.transition_graph = self.h5_file['graph'][()]
        self.shortest_path_distances = self.h5_file['shortest_path_distance'][()]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        self.initial_state_id = initial_state_id
        self.s_target = self._tiled_state(self.terminal_state_id)
        self.time = 0
        ###
        self.method = method
        # if self.method == 'word2vec_notarget' or self.method=='Transformer_word2vec_notarget'or self.method=='Transformer_word2vec_notarget_withposi'or self.method=='Transformer_word2vec_notarget_word2vec'or self.method=='Transformer_word2vec_notarget_word2vec_posi' or self.method=='gcn_transformer'or self.method =='Transformer_word2vec_notarget_word2vec_concat'or self.method == 'Transformer_word2vec_notarget_word2vec_action'or self.method =='grid_memory'or self.method=='Transformer_word2vec_notarget_action'or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=='grid_memory_action': #origin
        if self.method == 'word2vec_notarget' or self.method=='Transformer_word2vec_notarget'or self.method=='Transformer_word2vec_notarget_withposi'or self.method=='Transformer_word2vec_notarget_word2vec'or self.method=='Transformer_word2vec_notarget_word2vec_posi' or self.method=='gcn_transformer'or self.method =='Transformer_word2vec_notarget_word2vec_concat'or self.method == 'Transformer_word2vec_notarget_word2vec_action'or self.method =='grid_memory'or self.method =='grid_memory_no_observation'or self.method=='Transformer_word2vec_notarget_action'or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=='grid_memory_action'or self.method =='scene_only': #add
            self.scene = scene_name
            self.success = False
            self.bbox_method = bbox_method
            self.bbox_area = 0
            self.max_bbox_area = 0
            self.terminal_state = terminal_state
            self.object_vector = self.h5_file['object_vector']
            self.object_ids = json.loads(self.h5_file.attrs['object_ids'])
            self.s_target = self.object_vector[self.object_ids[self.terminal_state['object']]]
            self.mask_size = mask_size
            self.action_size = action_size
            self.last_action = -1
            self.object_visibility = [json.loads(j) for j in
                                      self.h5_file['object_visibility']]

    def _get_graph_handle(self):
        return self.h5_file

    def get_initial_states(self, method, goal):
        initial_states = list()
        # if method =='Transformer_word2vec_notarget'or self.method=='Transformer_word2vec_notarget_withposi'or self.method=='Transformer_word2vec_notarget_word2vec'or self.method=='Transformer_word2vec_notarget_word2vec_posi' or self.method=='gcn_transformer'or self.method =='Transformer_word2vec_notarget_word2vec_concat'or self.method == 'Transformer_word2vec_notarget_word2vec_action'or self.method =='grid_memory'or self.method=='Transformer_word2vec_notarget_action'or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=='grid_memory_action': #origin
        if method =='Transformer_word2vec_notarget'or self.method=='Transformer_word2vec_notarget_withposi'or self.method=='Transformer_word2vec_notarget_word2vec'or self.method=='Transformer_word2vec_notarget_word2vec_posi' or self.method=='gcn_transformer'or self.method =='Transformer_word2vec_notarget_word2vec_concat'or self.method == 'Transformer_word2vec_notarget_word2vec_action'or self.method =='grid_memory'or self.method =='grid_memory_no_observation'or self.method=='Transformer_word2vec_notarget_action'or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=='grid_memory_action'or self.method =='scene_only': #add
            while True:
                for k in range(self.n_locations):
                    min_d = self.shortest_path_terminal(k)
                    if min_d > 0:
                        initial_states.append(k)
                if len(initial_states) > 250:
                    break
        else:
            for k in range(self.n_locations):
                min_d = self.shortest_path_distances[k][goal]
                if min_d > 0:
                    initial_states.append(k)

        return initial_states

    @staticmethod
    def get_existing_initial_states(scene_name = 'bedroom_04',
        terminal_state_id = 0,
        h5_file_path = None):
        env = THORDiscreteEnvironment(h5_file_path, scene_name)
        return env.get_existing_initial_states(terminal_state_id)

    def reset(self, initial_state_id = None):
        # randomize initial state
        if initial_state_id is None:
            initial_state_id = self.initial_state_id
            #print("###")
            #print(initial_state_id)

        if initial_state_id is None:
            while True:
                k = random.randrange(self.n_locations)
                #print("@@@")
                #print(k)
                min_d = np.inf

                # check if target is reachable
                for t_state in self.terminal_states:
                    dist = self.shortest_path_distances[k][t_state]
                    min_d = min(min_d, dist)

                # min_d = 0  if k is a terminal state
                # min_d = -1 if no terminal state is reachable from k
                #if min_d > 0 and k%12 <= 3: break
                if min_d > 0: break
        else:
            k = initial_state_id
        
        # reset parameters
        self.current_state_id = k
        self.s_t = self._tiled_state(self.current_state_id)
        self.collided = False
        self.terminal = False
        self.bbox_area = 0
        self.max_bbox_area = 0
        self.time = 0
        self.success = False

    def step(self, action):
        assert not self.terminal, 'step() called in terminal state'
        k = self.current_state_id
        if self.actions[action] == "Done":
            self.last_action = action
            return
        if self.transition_graph[k][action] != -1:
            self.current_state_id = self.transition_graph[k][action]
            # if self.method != 'word2vec_notarget' and self.method!='Transformer_word2vec_notarget'and self.method!='Transformer_word2vec_notarget_withposi'and self.method!='Transformer_word2vec_notarget_word2vec'and self.method!='Transformer_word2vec_notarget_word2vec_posi' and self.method!='gcn_transformer'and self.method !='Transformer_word2vec_notarget_word2vec_concat'and self.method != 'Transformer_word2vec_notarget_word2vec_action'and self.method !='grid_memory'and self.method!='Transformer_word2vec_notarget_action'and self.method!='Transformer_word2vec_notarget_word2vec_action_posi'and self.method!='grid_memory_action': #origin
            if self.method != 'word2vec_notarget' and self.method!='Transformer_word2vec_notarget'and self.method!='Transformer_word2vec_notarget_withposi'and self.method!='Transformer_word2vec_notarget_word2vec'and self.method!='Transformer_word2vec_notarget_word2vec_posi' and self.method!='gcn_transformer'and self.method !='Transformer_word2vec_notarget_word2vec_concat'and self.method != 'Transformer_word2vec_notarget_word2vec_action'and self.method !='grid_memory'and self.method !='grid_memory_no_observation'and self.method!='Transformer_word2vec_notarget_action'and self.method!='Transformer_word2vec_notarget_word2vec_action_posi'and self.method!='grid_memory_action'or self.method =='scene_only': #add
                if self.terminals[self.current_state_id]:
                    self.terminal = True
                    self.collided = False
                else:
                    self.terminal = False
                    self.collided = False
        else:
            self.terminal = False
            self.collided = True

        self.s_t = np.append(self.s_t[:,1:], self._get_state(self.current_state_id), axis=1)
        self.time = self.time + 1
        self.last_action = action
        # if self.method == 'word2vec_notarget' or self.method=='Transformer_word2vec_notarget'or self.method=='Transformer_word2vec_notarget_withposi'or self.method=='Transformer_word2vec_notarget_word2vec'or self.method=='Transformer_word2vec_notarget_word2vec_posi' or self.method=='gcn_transformer'or self.method =='Transformer_word2vec_notarget_word2vec_concat'or self.method == 'Transformer_word2vec_notarget_word2vec_action'or self.method =='grid_memory'or self.method=='Transformer_word2vec_notarget_action'or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=='grid_memory_action': #origin
        if self.method == 'word2vec_notarget' or self.method=='Transformer_word2vec_notarget'or self.method=='Transformer_word2vec_notarget_withposi'or self.method=='Transformer_word2vec_notarget_word2vec'or self.method=='Transformer_word2vec_notarget_word2vec_posi' or self.method=='gcn_transformer'or self.method =='Transformer_word2vec_notarget_word2vec_concat'or self.method == 'Transformer_word2vec_notarget_word2vec_action'or self.method =='grid_memory'or self.method =='grid_memory_no_observation'or self.method=='Transformer_word2vec_notarget_action'or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=='grid_memory_action'or self.method =='scene_only': #add
            self.bbox_area = self._get_max_bbox_area(
                self.boudingbox, self.terminal_state['object'])
        #print(self.current_state_id)

    def _get_state(self, state_id):
        # read from hdf5 cache
        k = random.randrange(self.n_feat_per_location)
        return self.h5_file['resnet_feature'][state_id][k][:,np.newaxis]

    def _tiled_state(self, state_id):
        f = self._get_state(state_id)
        return np.tile(f, (1, self.history_length))

    def _get_max_bbox_area(self, bboxs, obj_class):
        area = 0
        for key, value in bboxs.items():
            keys = key.split('|')
            if keys[0] == obj_class:
                w = abs(value[0] - value[2])
                h = abs(value[1] + value[3])
                area = max(area, w * h)
        return area
    
    def _calculate_reward(self, terminal, collided, method):
        # positive reward upon task completion
        # if method == 'word2vec_notarget' or method == 'Transformer_word2vec_notarget'or self.method=='Transformer_word2vec_notarget_withposi'or self.method=='Transformer_word2vec_notarget_word2vec'or self.method=='Transformer_word2vec_notarget_word2vec_posi' or self.method=='gcn_transformer'or self.method =='Transformer_word2vec_notarget_word2vec_concat'or self.method == 'Transformer_word2vec_notarget_word2vec_action'or self.method =='grid_memory'or self.method=='Transformer_word2vec_notarget_action'or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=='grid_memory_action': #origin
        if method == 'word2vec_notarget' or method == 'Transformer_word2vec_notarget'or self.method=='Transformer_word2vec_notarget_withposi'or self.method=='Transformer_word2vec_notarget_word2vec'or self.method=='Transformer_word2vec_notarget_word2vec_posi' or self.method=='gcn_transformer'or self.method =='Transformer_word2vec_notarget_word2vec_concat'or self.method == 'Transformer_word2vec_notarget_word2vec_action'or self.method =='grid_memory'or self.method =='grid_memory_no_observation'or self.method=='Transformer_word2vec_notarget_action'or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=='grid_memory_action'or self.method =='scene_only': #add
            return self.reward_soft_goal()
        else:
            if terminal: return 10.0
            # time penalty or collision penalty
            #return -0.1 if collided else -0.01
            return -0.01

    def _calculate_bbox_reward(self):
        if self.bbox_area > self.max_bbox_area:
            self.max_bbox_area = self.bbox_area
            return self.bbox_area
        else:
            return 0
    
    def _downsample_bbox(self, input_shape, output_shape, input_bbox):
        h, w = input_shape
        out_h, out_w = output_shape
        # Between 0 and output_shape
        out_h = out_h - 1
        out_w = out_w - 1

        ratio_h = out_h / h
        ratio_w = out_w / w

        output = np.zeros(output_shape, dtype=np.float32)

        for i_bbox in input_bbox:
            bbox_xy, similarity = i_bbox
            x, y = bbox_xy
            out_x = int(x * ratio_w)
            out_y = int(y * ratio_h)
            output[out_x, out_y] = max(output[out_x, out_y], similarity)
        
        #print(output)
        return output

    @property
    def reward(self):
        return self._calculate_reward(self.is_terminal, self.collided, self.method)

    @property
    def is_terminal(self):
        return self.terminal or self.time >= 5000

    @property
    def boudingbox(self):
        # if self.bbox_method is None: #origin
        if self.bbox_method == 'bbox':
            # print("1) self.bbox_method : {}".format(self.bbox_method)) #test
            return json.loads(self.h5_file['bbox'][self.current_state_id])
        elif self.bbox_method == 'yolo':
            # print("2) self.bbox_method : {}".format(self.bbox_method)) #test
            return json.loads(self.h5_file['yolo_bbox'][self.current_state_id])
        elif self.bbox_method == 'vild':
            # print("2) self.bbox_method : {}".format(self.bbox_method)) #test
            return json.loads(self.h5_file['vild_bbox'][self.current_state_id])
        elif self.bbox_method == 'yolov7':
            # print("2) self.bbox_method : {}".format(self.bbox_method)) #test
            return json.loads(self.h5_file['yolov7_bbox'][self.current_state_id])


    def render(self, mode):
        if mode == 'resnet_features':
            return self.s_t
        elif mode == 'image':
            return self.h5_file['observation'][self.current_state_id]
        else:
            assert False

    def render_target(self, mode):
        if mode == 'resnet_features':
            return self.s_target
        elif mode == 'image':
            return self.h5_file['observation'][self.terminal_state_id]
        elif mode == 'word_features':
            return self.s_target
        else:
            assert False

    def render_mask_similarity(self):
        # Get shape of observation to downsample bbox location
        h, w, _ = np.shape(self.h5_file['observation'][0])
        #print(h)
        #print(w)
        bbox_location = []
        for key, value in self.boudingbox.items():
            keys = key.split('|')
            # Add bounding box if its the target object
            # if keys[0] == self.terminal_state['object']:
            # value[0] = start_x
            # value[2] = end_x
            x = value[0] + value[2]
            x = x/2

            # value[1] = start_y
            # value[3] = end_y
            y = value[1] + value[3]
            y = y/2

            # Ignore unknown ObjectId
            if keys[0] not in self.object_ids:
                continue

            curr_obj_id = self.object_ids[keys[0]]
            similarity = 1 - spatial.distance.cosine(
                self.s_target, self.object_vector[curr_obj_id])
            # for x in range(value[0], value[2], 1):
            #     for y in range(value[1], value[3], 1):
            # print("(type(h) and type(w)) : {}".format((isinstance(h, int)) and (isinstance(w, int)))) #add
            # print("x : {}, y : {}, similarity : {}".format(x, y, similarity)) #add
            # print("x : {}, y : {}, similarity : {}".format(x>0, y>0, similarity>0)) #add
            # if not np.isnan(x) and not np.isnan(y) and similarity>0:
            
            # latest(object detection)
            if not np.isnan(similarity) and similarity>0 and 0<=y<=h and 0<=x<=w : #(main) add
                bbox_location.append(((x, y), similarity))

            # pass(No object detection)
            # if self.bbox_method=="bbox" or self.bbox_method=="yolo":
            #     bbox_location.append(((x, y), similarity))
            # else:
            #     pass

        try:
            output = self._downsample_bbox(
                (h, w), (self.mask_size, self.mask_size), bbox_location)
            # print("ok : {}\n".format((h, w), (self.mask_size, self.mask_size), bbox_location)) #add
        except IndexError as e:
                                                                                                        
            print((h, w), bbox_location) #origin
            raise e
        return output[np.newaxis, np.newaxis, ...]

    def reward_soft_goal(self):
        GOAL_SUCCESS_REWARD = 5
        STEP_PENALTY = -0.01
        # BBOX area
        reward_ = self._calculate_bbox_reward()

        h, w, _ = np.shape(self.h5_file['observation'][0])

        # Normalize
        reward_ = reward_ / (h*w)

        # Use strict done
        # Emitted Done signal will trigger end of episode
        # Giving big reward only if object is visible
        if self.actions[self.last_action] == 'Done':
            self.success = False
            self.terminal = True
            # Check if object is visible
            for objectId in self.object_visibility[self.current_state_id]:
                obj = objectId.split('|')
                if obj[0] == self.terminal_state['object']:
                    #reward_ = reward_ + GOAL_SUCCESS_REWARD
                    reward_ = GOAL_SUCCESS_REWARD
                    self.success = True
                    #print(self.terminal_state['object'])
                    #print(objectId)
                    break
        else:
            reward_ = reward_ + STEP_PENALTY

        return reward_

    @property
    def actions(self):
        return ["MoveForward", "RotateRight", "RotateLeft", "MoveBackward","LookUp", "LookDown", "MoveRight", "MoveLeft","Done"]

    def shortest_path_terminal(self, state):
        # if self.method == 'Transformer_word2vec_notarget'or self.method=='Transformer_word2vec_notarget_withposi'or self.method=='Transformer_word2vec_notarget_word2vec'or self.method=='Transformer_word2vec_notarget_word2vec_posi' or self.method=='gcn_transformer'or self.method =='Transformer_word2vec_notarget_word2vec_concat'or self.method=='Transformer_word2vec_notarget_word2vec_action'or self.method =='grid_memory'or self.method=='Transformer_word2vec_notarget_action'or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=='grid_memory_action': #origin
        if self.method == 'Transformer_word2vec_notarget'or self.method=='Transformer_word2vec_notarget_withposi'or self.method=='Transformer_word2vec_notarget_word2vec'or self.method=='Transformer_word2vec_notarget_word2vec_posi' or self.method=='gcn_transformer'or self.method =='Transformer_word2vec_notarget_word2vec_concat'or self.method=='Transformer_word2vec_notarget_word2vec_action'or self.method =='grid_memory'or self.method =='grid_memory_no_observation'or self.method=='Transformer_word2vec_notarget_action'or self.method=='Transformer_word2vec_notarget_word2vec_action_posi'or self.method=='grid_memory_action'or self.method =='scene_only': #add
            lengths = []
            for i, object_visibility in enumerate(self.object_visibility):
                for objectId in object_visibility:
                    obj = objectId.split('|')
           
                    if obj[0] == self.terminal_state['object']:
                        if self.shortest_path_distances[state][i] != -1:
                            lengths.append(
                                self.shortest_path_distances[state][i])
                            break
            try:
         
                min_len = np.min(lengths)
            except Exception as e:
                print(self.scene, self.terminal_state)
                print(e)
                raise e
            return min_len
        else:
            return self.shortest_path_distances[state][self.terminal_state_id]

    @property
    def observation(self):
        return self.h5_file['observation'][self.current_state_id]
