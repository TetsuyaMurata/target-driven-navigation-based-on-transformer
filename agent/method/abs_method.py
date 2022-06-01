from abc import ABC, abstractmethod


class AbstractMethod(ABC):

    def __init__(self, method):
        super().__init__()
        self.method = method

    @abstractmethod
    def forward_policy(self, env, device, policy_networks):
        pass

    @abstractmethod
    def extract_input(self, env, device):
        pass
