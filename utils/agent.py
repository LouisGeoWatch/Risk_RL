# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeployNet(nn.Module):
    def __init__(self, nb_territories=6, hidden_size=16):
        super(DeployNet, self).__init__()

        self.input_fc = nn.Linear(nb_territories, hidden_size)
        self.output_fc = nn.Linear(hidden_size, nb_territories)

    def forward(self, reinforcements, player_presence_map):

        # Unsqueeze the inputs to add a batch dimension
        reinforcements, player_presence_map = reinforcements.unsqueeze(0), player_presence_map.unsqueeze(0)
        # Apply a linear layer to the input
        x = F.tanh(self.input_fc(player_presence_map))
        # Apply a linear layer to the output
        x = F.tanh(self.output_fc(x))

        # Find the possible actions (territories where the player has a presence)
        possible_actions = (player_presence_map > 0)
        # Mask the impossible actions
        mask = torch.zeros_like(x)
        mask[:, possible_actions] = 1
        x = torch.where(mask == 1, x, torch.zeros_like(x)-1000)

        # Softmax the output
        actions_prob = F.softmax(x, dim=1)
        return actions_prob


class AttackFortifyNet(nn.Module):
    def __init__(self, nb_territories=6, hidden_size=16):
        super(AttackFortifyNet, self).__init__()

        self.input_fc = nn.Linear(nb_territories, hidden_size)
        self.torig_fc = nn.Linear(hidden_size, nb_territories)
        self.tdest_fc = nn.Linear(hidden_size, nb_territories)

    def forward(self, possible_actions, player_presence_map):

        # Apply a linear layer to the input
        x = F.tanh(self.input_fc(player_presence_map))

        # Split the output into two vectors (one for t_orig, one for t_dest)
        torig = F.tanh(self.torig_fc(x))
        tdest = F.tanh(self.tdest_fc(x))

        # Build a matrix of output with a dot product
        embedding_mat = torch.outer(torig, tdest)
        # Put to zero all the positions that are not in possible_actions
        mask = torch.zeros_like(embedding_mat)
        for a in possible_actions:
            mask[a[0], a[1]] = 1
        embedding_mat = torch.where(mask == 1, embedding_mat, torch.zeros_like(embedding_mat)-1000)
        # Flatten the matrix
        embedding_mat = embedding_mat.flatten()
        embedding_mat = embedding_mat.unsqueeze(0)

        # Softmax the output
        actions_prob = F.softmax(embedding_mat, dim=1)

        return actions_prob


class PolicyGradientAgent():
    def __init__(self, nb_territories=6,
                 learning_rate=0.01, gamma=0.99,
                 deploy_hidden_size=16, attack_fortify_hidden_size=16):

        self.nb_territories = nb_territories
        self.gamma = gamma

        self.deploy_policy = DeployNet(nb_territories=nb_territories,
                                       hidden_size=deploy_hidden_size)
        self.attack_policy = AttackFortifyNet(nb_territories=nb_territories,
                                              hidden_size=attack_fortify_hidden_size)
        self.fortify_policy = AttackFortifyNet(nb_territories=nb_territories,
                                               hidden_size=attack_fortify_hidden_size)

        self.deploy_optimizer = torch.optim.Adam(self.deploy_policy.parameters(), lr=learning_rate)
        self.attack_optimizer = torch.optim.Adam(self.attack_policy.parameters(), lr=learning_rate)
        self.fortify_optimizer = torch.optim.Adam(self.fortify_policy.parameters(), lr=learning_rate)

    def choose_deploy_prob(self, reinforcements, player_presence_map):
        """Returns the territories to deploy troops on"""
        # Output of the deploy policy
        deploy_log_prob = self.deploy_policy(reinforcements, player_presence_map)
        # Sample an action
        m = Categorical(deploy_log_prob)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def choose_attack_prob(self, attacks, player_presence_map):
        """Returns the territory to attack"""
        # Output of the attack policy
        attack_log_prob = self.attack_policy(attacks, player_presence_map)
        # Sample an action
        m = Categorical(attack_log_prob)
        action = m.sample()
        # Decode the action into a territory pair
        t_orig = action // self.nb_territories
        t_dest = (action % self.nb_territories)
        return t_orig.item(), t_dest.item(), m.log_prob(action)

    def choose_fortify_prob(self, possible_fortifications, player_presence_map):
        """Returns the territory to fortify at the end of the turn"""
        # Output of the fortify policy
        fortify_log_prob = self.fortify_policy(possible_fortifications, player_presence_map)
        # Sample an action
        m = Categorical(fortify_log_prob)
        action = m.sample()
        # Decode the action into a territory pair
        t_orig = action // self.nb_territories
        t_dest = (action % self.nb_territories)
        return t_orig.item(), t_dest.item(), m.log_prob(action)

    def choose_deploy(self, reinforcements, player_presence_map):
        """The greedy version of choose_deploy_prob (takes the action with the highest probability)"""
        output = self.deploy_policy(reinforcements, player_presence_map)
        probs = F.softmax(output, dim=1).cpu()

        return torch.argmax(probs).item()

    def choose_attack(self, attacks, player_presence_map):
        """The greedy version of choose_attack_prob (takes the action with the highest probability)"""
        # Output of the attack policy
        attack_log_prob = self.attack_policy(attacks, player_presence_map)
        # Take the action with the highest probability
        action = torch.argmax(attack_log_prob)
        # Decode the action into a territory pair
        t_orig = action // self.nb_territories
        t_dest = (action % self.nb_territories)
        return t_orig.item(), t_dest.item()

    def choose_fortify(self, fortifications, player_presence_map):
        """The greedy version of choose_fortify_prob (takes the action with the highest probability)"""
        # Output of the fortify policy
        fortify_log_prob = self.fortify_policy(fortifications, player_presence_map)
        # Take the action with the highest probability
        action = torch.argmax(fortify_log_prob)
        # Decode the action into a territory pair
        t_orig = action // self.nb_territories
        t_dest = (action % self.nb_territories)
        return t_orig.item(), t_dest.item()

    def save(self, path):
        torch.save(self.deploy_policy.state_dict(), path + 'deploy_policy.pt')
        torch.save(self.attack_policy.state_dict(), path + 'attack_policy.pt')
        torch.save(self.fortify_policy.state_dict(), path + 'fortify_policy.pt')

    def load(self, path):
        self.deploy_policy.load_state_dict(torch.load(path + 'deploy_policy.pt'))
        self.attack_policy.load_state_dict(torch.load(path + 'attack_policy.pt'))
        self.fortify_policy.load_state_dict(torch.load(path + 'fortify_policy.pt'))


# We write a Naive Agent that always takes the first possible action among the possible ones
class NaiveAgent():
    def __init__(self, nb_territories=6):
        self.nb_territories = nb_territories

    def choose_deploy(self, reinforcements, player_presence_map):
        """Returns the territories to deploy troops on"""
        # Find the positive values in the presence map np array
        possible_actions = (player_presence_map > 0)
        # Find the indices of the positive values
        possible_actions = np.where(possible_actions)[0]
        # Take the first possible action
        return possible_actions[0]

    def choose_attack(self, attacks, player_presence_map):
        """Returns the territory to attack"""
        return attacks[0]

    def choose_fortify(self, fortifications, player_presence_map):
        """Returns the territory to fortify at the end of the turn"""
        return fortifications[0]


class RandomAgent():
    def __init__(self, nb_territories=6):
        self.nb_territories = nb_territories

    def choose_deploy(self, reinforcements, player_presence_map):
        """Returns the territories to deploy troops on"""
        # Find the positive values in the presence map np array
        possible_actions = (player_presence_map > 0)
        # Find the indices of the positive values
        possible_actions = np.where(possible_actions)[0]
        # Take a random possible action
        return np.random.choice(possible_actions)

    def choose_attack(self, attacks, player_presence_map):
        """Returns the territory to attack"""
        return attacks[np.random.randint(len(attacks))]

    def choose_fortify(self, fortifications, player_presence_map):
        """Returns the territory to fortify at the end of the turn"""
        return fortifications[np.random.randint(len(fortifications))]


class GengisKhan():
    def __init__(self, nb_territories=6):
        self.nb_territories = nb_territories

    def choose_deploy(self, reinforcements, player_presence_map):
        """Returns the territories to deploy troops on"""
        # Always deploy on the strongest territory
        return np.argmax(player_presence_map)

    def choose_attack(self, attacks, player_presence_map):
        """Returns the territory to attack"""
        # Attacks with the most powerful territory available
        available_troops = [player_presence_map[e[0]] for e in attacks]
        idx_max = np.argmax(available_troops)
        return attacks[idx_max]

    def choose_fortify(self, fortifications, player_presence_map):
        """Returns the territory to fortify at the end of the turn"""
        # Fortify a territory with the most troops possible
        troop_presence = [player_presence_map[start] + player_presence_map[end] - 1
                          for (start, end) in fortifications]
        idx_max = np.argmax(troop_presence)
        return fortifications[idx_max]


class Courage():
    def __init__(self, nb_territories=6):
        self.nb_territories = nb_territories

    def choose_deploy(self, reinforcements, player_presence_map):
        """Returns the territories to deploy troops on"""
        # Find the positive values in the presence map np array
        possible_actions = (player_presence_map > 0)
        # Find the indices of the positive values
        possible_actions = np.where(possible_actions)[0]
        # Find the number of troops on each territory
        available_troops = np.array([[t, player_presence_map[t]] for t in possible_actions])
        # Choose the weakest territory
        weakest = np.amin(available_troops, axis=0)[1]
        return player_presence_map.index(weakest)

    def choose_attack(self, attacks, player_presence_map):
        """Returns the territory to attack"""
        # Attacks the weakest troops with its strongest troop
        delta_troops = [player_presence_map[start] + player_presence_map[end]
                        for (start, end) in attacks]
        return attacks[np.argmax(delta_troops)]

    def choose_fortify(self, fortifications, player_presence_map):
        """Returns the territory to fortify at the end of the turn"""
        # Always fortify from the strongest territory possible to the weakest possible
        troop_presence = [player_presence_map[start] - player_presence_map[end]
                          for (start, end) in fortifications]
        idx_max = np.argmax(troop_presence)
        return fortifications[idx_max]
