# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyGradientAgent():
    def __init__(self, presence_map_shape, nb_territories):
        self.deploy_policy = nn.Sequential(
                                        nn.Linear(presence_map_shape+1, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, nb_territories)
                                        )

        self.attack_policy = nn.Sequential(
                                        nn.Linear(presence_map_shape+1, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, nb_territories*2),
                                        nn.ReLU()
                                        )

        self.fortify_policy = nn.Sequential(
                                        nn.Linear(presence_map_shape+1, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, nb_territories*2),
                                        nn.ReLU()
                                        )

    def choose_deploy_prob(self, reinforcements, world):
        """Returns the territories to deploy troops on"""

        input = torch.cat((reinforcements, world.presence_map.flatten()), 0).to(device)
        probs = F.softmax(self.deploy_policy(input), dim=1).cpu()
        # To categorical
        m = Categorical(probs)
        # Sample an action from the probability distribution of the output
        t = m.sample()

        return t.item(), reinforcements, m.log_prob(t)

    def choose_attack_prob(self, attack_outcomes, world):
        """Returns the territory to attack"""

        input = torch.cat((attack_outcomes, world.presence_map.flatten()), 0).to(device)
        probs = F.softmax(self.attack_policy(input), dim=1).cpu()
        # To categorical
        m = Categorical(probs)
        # Sample an action from the probability distribution of the output
        t = m.sample()

        return t.item(), attack_outcomes, m.log_prob(t)

    def choose_conquest(self, world):
        """Returns the number of troops transfered
           to the conquered territory"""
        pass

    def choose_fortify_prob(self, fortifications, world):
        """Returns the territory to fortify at the end of the turn"""

        input = torch.cat((fortifications, world.presence_map.flatten()), 0).to(device)
        probs = F.softmax(self.fortify_policy(input), dim=1).cpu()
        # To categorical
        m = Categorical(probs)
        # Sample an action from the probability distribution of the output
        t = m.sample()

        return t.item(), fortifications, m.log_prob(t)


class Human_player():
    def __init__(self):
        pass

    def choose_deploy(self):
        pass

    def choose_attack(self):
        pass

    def choose_fortify(self):
        pass
