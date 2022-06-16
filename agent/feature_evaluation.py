

import csv
import imp
import logging
import os
import random
import sys
from itertools import groupby

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from tensorboardX import SummaryWriter

from agent.environment.ai2thor_file import \
    THORDiscreteEnvironment as THORDiscreteEnvironmentFile
from agent.gpu_thread import GPUThread
from agent.method.aop import AOP
from agent.method.gcn import GCN
from agent.method.similarity_grid import SimilarityGrid
from agent.method.target_driven import TargetDriven
from agent.network import SceneSpecificNetwork, SharedNetwork
from agent.training import TrainingSaver
from agent.utils import find_restore_points, get_first_free_gpu
from torchvision import transforms

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


class FeatureEvaluation:
    def __init__(self, config):
        self.config = config
        self.method = config['method']
        gpu_id = get_first_free_gpu(2000)
        self.device = torch.device("cuda:" + str(gpu_id))
        if self.method != "random":
            self.shared_net = SharedNetwork(
                self.config['method'], self.config.get('mask_size', 5)).to(self.device)
            self.scene_net = SceneSpecificNetwork(
                self.config['action_size']).to(self.device)

        self.checkpoints = []
        self.checkpoint_id = 0
        self.saver = None
        self.chk_numbers = None

    @staticmethod
    def load_checkpoints(config, fail=True):
        evaluation = FeatureEvaluation(config)
        checkpoint_path = config.get(
            'checkpoint_path', 'model/checkpoint-{checkpoint}.pth')

        checkpoints = []
        (base_name, chk_numbers) = find_restore_points(checkpoint_path, fail)
        if evaluation.method != "random":
            try:
                for chk_name in base_name:
                    state = torch.load(
                        open(os.path.join(os.path.dirname(checkpoint_path), chk_name), 'rb'))
                    checkpoints.append(state)
            except Exception as e:
                print("Error loading", e)
                exit()
            evaluation.saver = TrainingSaver(evaluation.shared_net,
                                             evaluation.scene_net, None, evaluation.config)
        evaluation.chk_numbers = chk_numbers
        evaluation.checkpoints = checkpoints
        return evaluation

    def restore(self):
        print('Restoring from checkpoint',
              self.chk_numbers[self.checkpoint_id])
        self.saver.restore(self.checkpoints[self.checkpoint_id])

    def next_checkpoint(self):
        self.checkpoint_id = (self.checkpoint_id + 1) % len(self.checkpoints)

    def run(self):
        random.seed(200)
        num_episode_eval = 10

        self.method_class = None
        if self.method == 'word2vec' or self.method == 'word2vec_noconv' or self.method == 'word2vec_notarget' or self.method == 'word2vec_nosimi':
            self.method_class = SimilarityGrid(self.method)
        elif self.method == 'aop' or self.method == 'aop_we':
            self.method_class = AOP(self.method)
        elif self.method == 'target_driven':
            self.method_class = TargetDriven(self.method)
        elif self.method == 'gcn':
            self.method_class = GCN(self.method)

        for chk_id in self.chk_numbers:
            scene_stats = dict()
            for scene_scope, items in self.config['task_list'].items():
                self.restore()
                scene_stats[scene_scope] = dict()
                scene_stats[scene_scope]["length"] = list()
                scene_stats[scene_scope]["spl"] = list()
                scene_stats[scene_scope]["success"] = list()
                scene_stats[scene_scope]["spl_long"] = list()
                scene_stats[scene_scope]["success_long"] = list()

                for task_scope in items:
                    env = THORDiscreteEnvironmentFile(scene_name=scene_scope,
                                                      method=self.method,
                                                      reward=self.config['reward'],
                                                      h5_file_path=(lambda scene: self.config.get(
                                                          "h5_file_path").replace('{scene}', scene)),
                                                      terminal_state=task_scope,
                                                      action_size=self.config['action_size'],
                                                      mask_size=self.config.get(
                                                          'mask_size', 5))
                    print("Current task:", env.terminal_state['object'])
                    for i_episode in range(num_episode_eval):
                        if not env.reset():
                            continue
                        ep_t = 0
                        terminal = False
                        while not terminal:

                            if self.method != "random":
                                policy, value, state = self.method_class.forward_policy(
                                    env, self.device, lambda x: self.scene_net(self.shared_net(x)))
                                policy_softmax = F.softmax(policy, dim=0)
                                action = policy_softmax.multinomial(
                                    1).data.cpu().numpy()[0]

                            env.step(action)
                            if ep_t == 500:
                                terminal = True
                                break
                            ep_t += 1
                            env.reward
                            terminal = env.terminal
                            # Compute CAM only for terminal state
                            if terminal and env.success:
                                # Retrieve the feature from the convolution layer (similarity grid)
                                conv_output = self.shared_net.net.conv_output

                                state, x_processed, object_mask = self.method_class.extract_input(
                                    env, self.device)

                                # Create one hot vector for outputted action
                                one_hot_vector = torch.zeros(
                                    (1, env.action_size), dtype=torch.float32)
                                one_hot_vector[0][action] = 500
                                one_hot_vector = one_hot_vector.to(self.device)

                                # Reset grad
                                self.shared_net.zero_grad()
                                self.scene_net.zero_grad()

                                # Backward pass with specified action
                                policy.backward(gradient=one_hot_vector, retain_graph=True)

                                # Get hooked gradients for CAM
                                guided_gradients = self.shared_net.net.gradient.cpu().data.numpy()[
                                    0]

                                # Get hooked gradients for Vanilla
                                vanilla_grad = self.shared_net.net.gradient_vanilla.cpu()
                                vanilla_grad = vanilla_grad.data.numpy()[0]

                                # Get convolution outputs
                                target = conv_output.cpu().data.numpy()[0]

                                # Get weights from gradients
                                # Take averages for each gradient
                                weights = np.mean(guided_gradients, axis=(1, 2))
                                # Create empty numpy array for cam
                                cam = np.ones(target.shape[1:], dtype=np.float32)

                                # Multiply each weight with its conv output and then, sum
                                for i, w in enumerate(weights):
                                    cam += w * target[i, :, :]
                                cam = np.maximum(cam, 0)
                                cam = (cam - np.min(cam)) / (np.max(cam) -
                                                             np.min(cam))  # Normalize between 0-1
                                cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
                                cam = np.uint8(Image.fromarray(cam).resize((object_mask.shape[2],
                                                                            object_mask.shape[3]), Image.ANTIALIAS))/255

                                # Create vanilla saliency img
                                vanilla_grad = vanilla_grad - vanilla_grad.min()
                                vanilla_grad /= vanilla_grad.max()

                                fig = plt.figure(figsize=(7*1.5, 2*1.5))
                                obs_plt = fig.add_subplot(141)
                                simi_grid_plt = fig.add_subplot(142)
                                cam_plt = fig.add_subplot(143)
                                vanilla_plt = fig.add_subplot(144)

                                # Observation visualization
                                obs_plt.title.set_text('Observation, Target:' +
                                                       env.terminal_state['object'])
                                obs_plt.imshow(env.observation)

                                # Simliratity grid visualization
                                simi_grid_plt.title.set_text("Similarity grid")
                                ob_mask_viz = object_mask.cpu().squeeze()
                                ob_mask_viz = np.flip(np.rot90(ob_mask_viz), axis=0)
                                simi_grid_plt.imshow(ob_mask_viz,
                                                     vmin=0, vmax=1, cmap='gray')

                                # CAM visualisation
                                cam_plt.title.set_text("CAM visualization")
                                cam_viz = np.flip(np.rot90(cam), axis=0)
                                cam_plt.imshow(cam_viz, vmin=0, vmax=1, cmap='plasma')

                                # Vanilla saliency visualization
                                vanilla_grad = vanilla_grad.squeeze(0)
                                van_viz = np.uint8(Image.fromarray(vanilla_grad).resize((object_mask.shape[2],
                                                                                         object_mask.shape[3]), Image.ANTIALIAS))/255
                                vanilla_plt.title.set_text("Vanilla saliency visualization")
                                van_viz = np.flip(np.rot90(vanilla_grad), axis=0)
                                vanilla_plt.imshow(van_viz, vmin=0, vmax=1, cmap='gray')

                                plt.tight_layout()
                                plt.show()

            break


'''
# Load weights trained on tensorflow
data = pickle.load(
    open(os.path.join(__file__, '..\\..\\weights.p'), 'rb'), encoding='latin1')
def convertToStateDict(data):
    return {key:torch.Tensor(v) for (key, v) in data.items()}

shared_net.load_state_dict(convertToStateDict(data['navigation']))
for key in TASK_LIST.keys():
    scene_nets[key].load_state_dict(convertToStateDict(data[f'navigation/{key}']))'''
