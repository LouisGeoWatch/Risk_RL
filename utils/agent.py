# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyGradientAgent():
    def __init__(self, nb_players=3, nb_territories=6,
                 learning_rate=0.01, gamma=0.99):

        presence_map_shape = nb_players*nb_territories

        self.deploy_policy = nn.Sequential(
                                        nn.Linear(presence_map_shape+1, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, nb_territories)
                                        )

        self.attack_policy = nn.Sequential(
                                        nn.Linear(presence_map_shape+1, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 1),
                                        nn.ReLU()
                                        )

        self.fortify_policy = nn.Sequential(
                                        nn.Linear(presence_map_shape+1, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, nb_territories*2),
                                        nn.ReLU()
                                        )

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = gamma

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

        # attack_outcomes is a list of tuples (t, a, prob). We need to convert it to a tensor
        # as if it were a batch of data
        attack_outcomes = torch.tensor(attack_outcomes).to(device)
        # Now concatenate the flattened presence map to the attack_outcomes tensor
        input = torch.cat((attack_outcomes,
                           world.presence_map.flatten().repeat(attack_outcomes.shape[0], 1)), 1).to(device)
        # Pass the input through the network
        output = self.attack_policy(input)
        # Apply the softmax function to the output to get the probabilities
        probs = F.softmax(output, dim=0).cpu()
        # To categorical
        m = Categorical(probs)
        # Sample an action from the probability distribution of the output
        t = m.sample()

        return t.item(), attack_outcomes[t.item()], m.log_prob(t)

    def choose_conquest(self, world):
        """Returns the number of troops transfered
           to the conquered territory"""
        pass

    def choose_fortify_prob(self, fortifications, world):
        """Returns the territory to fortify at the end of the turn"""

        # Same as choose_attack_prob
        fortifications = torch.tensor(fortifications).to(device)
        input = torch.cat((fortifications,
                           world.presence_map.flatten().repeat(fortifications.shape[0], 1)), 1).to(device)
        output = self.fortify_policy(input)
        probs = F.softmax(output, dim=0).cpu()
        # To categorical
        m = Categorical(probs)
        # Sample an action from the probability distribution of the output
        t = m.sample()

        return t.item(), fortifications[t.item()], m.log_prob(t)

    def choose_fortify(self, fortifications, world):
        """The greedy version of choose_fortify_prob"""
        fortifications = torch.tensor(fortifications).to(device)
        input = torch.cat((fortifications,
                           world.presence_map.flatten().repeat(fortifications.shape[0], 1)), 1).to(device)

        output = self.fortify_policy(input)
        probs = F.softmax(output, dim=0).cpu()

        return torch.argmax(probs).item(), fortifications[torch.argmax(probs).item()]

    def choose_attack(self, attack_outcomes, world):
        """The greedy version of choose_attack_prob"""
        attack_outcomes = torch.tensor(attack_outcomes).to(device)
        input = torch.cat((attack_outcomes,
                           world.presence_map.flatten().repeat(attack_outcomes.shape[0], 1)), 1).to(device)
        output = self.attack_policy(input)
        probs = F.softmax(output, dim=0).cpu()

        return torch.argmax(probs).item(), attack_outcomes[torch.argmax(probs).item()]

    def choose_deploy(self, reinforcements, world):
        """The greedy version of choose_deploy_prob"""
        input = torch.cat((reinforcements, world.presence_map.flatten()), 0).to(device)
        probs = F.softmax(self.deploy_policy(input), dim=1).cpu()

        return torch.argmax(probs).item(), reinforcements


class Human_player():
    def __init__(self):
        pass

    def choose_deploy(self):
        pass

    def choose_attack(self):
        pass

    def choose_fortify(self):
        pass
