# -*- coding: utf-8 -*-
import json
import random

import torch
import h5py
import numpy as np
from scipy import spatial

from agent.environment.environment import Environment


class THORDiscreteEnvironment(Environment):

    acts = ["MoveAhead", "RotateRight", "RotateLeft", "MoveBack",
            "LookUp", "LookDown", "MoveRight", "MoveLeft", "Done"]

    def __init__(self,
                 method: str,
                 reward: str,
                 scene_name='FloorPlan1',
                 n_feat_per_location=1,
                 history_length: int = 4,
                 terminal_state=0,
                 h5_file_path=None,
                 bbox_method=None,
                 we_method=None,
                 action_size: int = 4,
                 mask_size: int = 5,
                 **kwargs):
        """THORDiscreteEnvironment constructor, it represent a world where an agent evolves

        Keyword Arguments:
            scene_name {str} -- Name of the current world (default: {'bedroom_04'})
            n_feat_per_location {int} -- Number of feature by position in the world (default: {1})
            history_length {int} -- Number of frame to stack so the network take in account previous observations (default: {4})
            terminal_state_id {int} -- Terminal position represented by an ID (default: {0})
            h5_file_path {[type]} -- Path to precomputed world (default: {None})
        """
        super(THORDiscreteEnvironment, self).__init__()

        # Load dataset name for this scene
        if h5_file_path is None:
            h5_file_path = f"/app/data/{scene_name}.h5"
        elif callable(h5_file_path):
            h5_file_path = h5_file_path(scene_name)

        self.scene = scene_name

        # Store terminal state
        self.terminal_state = terminal_state

        # Load dataset
        self.h5_file = h5py.File(h5_file_path, 'r')

        # Number of resnet feature per location (1)
        self.n_feat_per_location = n_feat_per_location

        # Number of stacked frame
        self.history_length = history_length

        self.locations = self.h5_file['location'][()]
        self.rotations = self.h5_file['rotation'][()]
        self.n_locations = self.locations.shape[0]

        # State action graph
        self.transition_graph = self.h5_file['graph'][()]

        # Number of possible action
        self.action_size = action_size

        # Type of method used (word2vec, aop or target_driven)
        self.method = method

        # Type of reward fun (bbox or step)
        self.reward_fun = reward

        # Load object id dict
        self.object_ids = json.loads(self.h5_file.attrs['object_ids'])

        # Load object resnet feature
        object_feature = self.h5_file['object_feature']

        # Save word embedding method (None is spacy en_core_web_lg, visualgenome is we trained on visual genome)
        self.we_method = we_method

        # Load object word embedding feature
        if self.we_method is None:
            self.object_vector = self.h5_file['object_vector']
        else:
            self.object_vector = self.h5_file['object_vector_visualgenome']

        # Load shortest path distance between state
        self.shortest_path_distance = self.h5_file['shortest_path_distance']

        # Load object visibility
        self.object_visibility = [json.loads(j) for j in
                                  self.h5_file['object_visibility']]

        # Save bbox method (None is groundtruth, yolo is yolo bbox)
        self.bbox_method = bbox_method

        self.bbox_area = 0
        self.max_bbox_area = 0

        self.time = 0

        self.terminal_id = -1

        self.last_action = -1

        self.success = False

        self.shortest_path_threshold = 5

        if self.reward_fun == 'soft_goal':
            if "Done" not in self.acts[:self.action_size]:
                print("ERROR: Done action need to be used with soft goal")
                exit()
        elif self.reward_fun == 'env_goal':
            pass

        else:
            terminal_pos = list(self.terminal_state['position'].values())
            for term_id, loc in enumerate(self.locations):
                if np.array_equal(loc, terminal_pos):
                    self.terminal_id = term_id
                    break

        # LAST instruction
        if self.method == 'word2vec' or self.method == 'word2vec_nosimi' or \
           self.method == 'word2vec_noconv' or self.method == 'word2vec_notarget' or \
           self.method == 'gcn' or self.method == 'aop_we' or self.method == 'word2vec_notarget_lstm' or \
           self.method == 'word2vec_notarget_lstm_2layer' or self.method == 'word2vec_notarget_lstm_3layer' or \
           self.method == 'word2vec_notarget_rnn' or self.method == 'word2vec_notarget_gru':
            self.s_target = self.object_vector[self.object_ids[self.terminal_state['object']]]

        elif self.method == 'aop':
            self.s_target = object_feature[self.object_ids[self.terminal_state['object']]]

        elif self.method == 'target_driven':
            # LAST instruction
            terminal_id = None
            for i, loc in enumerate(self.locations):
                if np.array_equal(loc, list(self.terminal_state['position'].values())):
                    if np.array_equal(self.rotations[i], list(self.terminal_state['rotation'].values())):
                        terminal_id = i
                        break
            self.s_target = self._tiled_state(terminal_id)
        elif self.method == "random":
            pass
        else:
            raise Exception('Please choose a method')

        self.mask_size = mask_size

    def reset(self, set_state=True):
        # randomize initial state
        if set_state:
            ks = np.arange(0, self.n_locations)
            random.shuffle(ks)
            k_set = False
            k_final = None
            while not k_set:
                for k in ks:
                    # Assure that Z value is 0
                    if self.rotations[k][2] == 0:
                        # Assure that shortest path is higher than 0 (not starting in final state)
                        if self.accessible_terminal(k) and self.shortest_path_terminal(k) > 0:
                            k_set = True
                            k_final = k
                            break
                if not k_set:
                    print(self.scene, 'Did not find accessible state for',
                          self.terminal_state['object'])
                    return False
            # reset parameters
            self.current_state_id = k_final
            self.start_state_id = k_final
        if self.method != "random":
            self.s_t = self._tiled_state(self.current_state_id)
        self.collided = False
        self.terminal = False
        self.bbox_area = 0
        self.max_bbox_area = 0
        self.time = 0
        self.success = False
        if self.method == 'word2vec_notarget_lstm_2layer':
            self.hidden_state = (torch.zeros(2, 1, 512), torch.zeros(2, 1, 512))
        elif self.method == 'word2vec_notarget_lstm_3layer':
            self.hidden_state = (torch.zeros(3, 1, 512), torch.zeros(3, 1, 512))
        elif self.method == 'word2vec_notarget_lstm':
            self.hidden_state = (torch.zeros(1, 1, 512), torch.zeros(1, 1, 512))
        else:
            self.hidden_state = torch.zeros(1, 1, 512)
        return True

    def step(self, action):
        assert not self.terminal, 'step() called in terminal state'
        k = self.current_state_id
        if self.acts[action] == "Done":
            self.last_action = action
            return
        if self.transition_graph[k][action] != -1:
            self.current_state_id = self.transition_graph[k][action]
            if self.reward_fun == 'env_goal':
                for objectId in self.object_visibility[self.current_state_id]:
                    obj = objectId.split('|')
                    if obj[0] == self.terminal_state['object']:
                        self.terminal = True
                        self.success = True
            elif self.reward_fun != "soft_goal":
                agent_pos = self.locations[self.current_state_id]  # NDARRAY
                # Check only y value
                agent_rot = self.rotations[self.current_state_id][1]

                terminal_pos = list(
                    self.terminal_state['position'].values())  # NDARRAY
                # Check only y value
                terminal_rot = self.terminal_state['rotation']['y']

                if np.array_equal(agent_pos, terminal_pos) and np.array_equal(agent_rot, terminal_rot):
                    self.terminal = True
                    self.success = True
                    self.collided = False
                else:
                    self.terminal = False
                    self.collided = False

        else:
            self.terminal = False
            self.collided = True

        if self.method != "random":
            self.s_t = np.append(self.s_t[:, 1:], self._get_state(
                self.current_state_id), axis=1)

        # Retrieve bounding box area of target object class
        self.bbox_area = self._get_max_bbox_area(
            self.boudingbox, self.terminal_state['object'])

        self.time = self.time + 1
        self.last_action = action

    def _get_state(self, state_id):
        # read from hdf5 cache
        k = random.randrange(self.n_feat_per_location)
        return self.h5_file['resnet_feature'][state_id][k][:, np.newaxis]

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
        return output

    @property
    def reward(self):
        if self.reward_fun == 'bbox':
            reward_ = self._calculate_bbox_reward()
            return reward_

        elif self.reward_fun == 'step':
            return 10.0 if self.terminal else -0.01
        elif self.reward_fun == 'soft_goal':
            return self.reward_soft_goal()
        elif self.reward_fun == 'env_goal':
            return self.reward_env_goal()

    @property
    def is_terminal(self):
        return self.terminal or self.time >= 200

    @property
    def observation(self):
        return self.h5_file['observation'][self.current_state_id]

    @property
    def boudingbox(self):
        if self.bbox_method is None:
            return json.loads(self.h5_file['bbox'][self.current_state_id])
        elif self.bbox_method == 'yolo':
            return json.loads(self.h5_file['yolo_bbox'][self.current_state_id])

    def render(self, mode):
        assert mode == 'resnet_features'
        return self.s_t

    def render_target(self, mode):
        if self.method == 'aop' or self.method == 'aop_we' or self.method == 'word2vec' or self.method == 'word2vec_nosimi' or self.method == 'word2vec_noconv' or self.method == "gcn":
            assert mode == 'word_features'
            return self.s_target
        elif self.method == 'target_driven':
            assert mode == 'resnet_features'
            return self.s_target

    def render_mask_similarity(self):
        # Get shape of observation to downsample bbox location
        h, w, _ = np.shape(self.h5_file['observation'][0])

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
            bbox_location.append(((x, y), similarity))
        try:
            output = self._downsample_bbox(
                (h, w), (self.mask_size, self.mask_size), bbox_location)
        except IndexError as e:
            print((h, w), bbox_location)
            raise e
        return output[np.newaxis, np.newaxis, ...]

    def render_mask(self):
        # Get shape of observation to downsample bbox location
        h, w, _ = np.shape(self.h5_file['observation'][0])

        bbox_location = []
        for key, value in self.boudingbox.items():
            keys = key.split('|')
            if keys[0] == self.terminal_state['object']:
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
                bbox_location.append(((x, y), 1))
        try:
            output = self._downsample_bbox(
                (h, w), (self.mask_size, self.mask_size), bbox_location)
        except IndexError as e:
            print((h, w), bbox_location)
            raise e
        return output

    def accessible_terminal(self, state):

        if self.reward_fun == 'soft_goal' or self.reward_fun == 'env_goal':
            lengths = []
            for i, object_visibility in enumerate(self.object_visibility):
                for objectId in object_visibility:
                    obj = objectId.split('|')
                    if obj[0] == self.terminal_state['object']:
                        if self.shortest_path_distance[state][i] != -1:
                            return True
            return False
        else:
            return self.shortest_path_distance[state][self.terminal_id] != -1

    def shortest_path_terminal(self, state):

        if self.reward_fun == 'soft_goal' or self.reward_fun == 'env_goal':
            lengths = []
            for i, object_visibility in enumerate(self.object_visibility):
                for objectId in object_visibility:
                    obj = objectId.split('|')
                    if obj[0] == self.terminal_state['object']:
                        if self.shortest_path_distance[state][i] != -1:
                            lengths.append(
                                self.shortest_path_distance[state][i])
                            break
            try:
                min_len = np.min(lengths)
            except Exception as e:
                print(self.scene, self.terminal_state)
                print(e)
                raise e
            return min_len
        else:
            return self.shortest_path_distance[state][self.terminal_id]

    @property
    def actions(self):
        return self.acts[: self.action_size]

    def stop(self):
        pass

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
        if self.acts[self.last_action] == 'Done':
            self.success = False
            self.terminal = True
            # Check if object is visible
            for objectId in self.object_visibility[self.current_state_id]:
                obj = objectId.split('|')
                if obj[0] == self.terminal_state['object']:
                    reward_ = reward_ + GOAL_SUCCESS_REWARD
                    self.success = True
                    break
        else:
            reward_ = reward_ + STEP_PENALTY

        return reward_

    def reward_env_goal(self):
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
        # Check if object is visible
        for objectId in self.object_visibility[self.current_state_id]:
            obj = objectId.split('|')
            if obj[0] == self.terminal_state['object']:
                reward_ = reward_ + GOAL_SUCCESS_REWARD
                self.success = True
                break
        else:
            reward_ = reward_ + STEP_PENALTY

        return reward_

    def set_hidden(self, hidden):
        self.hidden_state = hidden

    def render_hidden_state(self):
        return self.hidden_state