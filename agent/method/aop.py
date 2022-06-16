import torch

from .abs_method import AbstractMethod


class AOP(AbstractMethod):
    def extract_input(self, env, device):
        state = {
            "current": env.render('resnet_features'),
            "goal": env.render_target('word_features'),
            "object_mask": env.render_mask()
        }
        x_processed = torch.from_numpy(state["current"])
        goal_processed = torch.from_numpy(state["goal"])
        object_mask = torch.from_numpy(state['object_mask'])

        x_processed = x_processed.to(device)
        goal_processed = goal_processed.to(device)
        object_mask = object_mask.to(device)

        return state, x_processed, goal_processed, object_mask

    def forward_policy(self, env, device, policy_networks):

        state, x_processed, goal_processed, object_mask = self.extract_input(env, device)

        if self.method == 'aop' or self.method == 'aop_we':
            (policy, value) = policy_networks(
                (x_processed, goal_processed, object_mask,))
        return policy, value, state
