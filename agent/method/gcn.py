import torch

from torchvision import transforms

from .abs_method import AbstractMethod


class GCN(AbstractMethod):
    def extract_input(self, env, device):
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])])
        state = {
            "current": env.render('resnet_features'),
            "goal": env.render_target('word_features'),
            "observation": normalize(env.observation).unsqueeze(0),
        }

        x_processed = torch.from_numpy(state["current"])
        goal_processed = torch.from_numpy(state["goal"])
        obs = state['observation']

        x_processed = x_processed.to(device)
        goal_processed = goal_processed.to(device)
        obs = obs.to(device)

        return state, x_processed, goal_processed, obs

    def forward_policy(self, env, device, policy_networks):

        state, x_processed, goal_processed, obs = self.extract_input(env, device)

        (policy, value) = policy_networks(
            (x_processed, goal_processed, obs,))

        return policy, value, state
