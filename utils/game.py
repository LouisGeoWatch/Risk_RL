import copy
from collections import deque
import numpy as np
from tqdm import tqdm
import torch

from utils.world import World
from utils.agent import PolicyGradientAgent
from utils.viz import draw_map, draw_map_and_save

# World Map
map_graph = np.array([
    [0, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0]
])

# Countries names
countries = {0: "North America",
             1: "South America",
             2: "Europe",
             3: "Africa",
             4: "Asia",
             5: "South-Asia / Oceania"
             }

# Nb player
nb_player = 3

# Vector of presence on the map
# P[i, j] = #troops of player i in zone j
presence_map = np.array([
    [2, 2, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2]
])

layout = {0: np.array([0.5, 0.5]),
          1: np.array([0.5, 0.3]),
          2: np.array([0.5, 0.7]),
          3: np.array([0.7, 0.5]),
          4: np.array([0.3, 0.5]),
          5: np.array([0.3, 0.3])
          }

colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow',
          4: 'purple', 5: 'orange', 6: 'black', 7: 'white'}

label_options = {"ec": "k", "fc": "white", "alpha": 0.3}


# Probabilities of winning
def pb_table(N=50):
    dp = [[0.]*(N+1) for _ in range(N+1)]  # (i, j) = defender with i armies, attacker with j armies
    dp[0][1:] = [1.]*N
    for i in range(1, N+1):
        for j in range(1, N+1):
            if i == 1:
                if j == 1:
                    dp[i][j] = 15/36 * dp[i-1][j] + 21/36 * dp[i][j-1]
                elif j == 2:
                    dp[i][j] = 125/216 * dp[i-1][j] + 91/216 * dp[i][j-1]
                else:
                    dp[i][j] = 855/1296 * dp[i-1][j] + 441/1296 * dp[i][j-1]
            else:
                if j == 1:
                    dp[i][j] = 55/216 * dp[i-1][j] + 161/216 * dp[i][j-1]
                elif j == 2:
                    dp[i][j] = 295/1296 * dp[i-2][j] + 420/1296 * dp[i-1][j-1] + 581/1296 * dp[i][j-2]
                else:
                    dp[i][j] = 2890/7776 * dp[i-2][j] + 2611/7776 * dp[i-1][j-1] + 2275/7776 * dp[i][j-2]
    return dp


proba_table = np.array(pb_table()).T


class Game():
    def __init__(self, map_graph=map_graph, presence_map=presence_map,
                 nb_players=nb_player, proba_table=proba_table,
                 reinforcement_param=3,
                 countries=countries, colors=colors, layout=layout,
                 label_options=label_options):

        self.map_graph = map_graph.copy()
        self.presence_map = presence_map.copy()
        self.players = nb_players
        self.proba_table = proba_table
        self.cur_turn = 0
        self.game_over = False
        self.reinforcement_param = reinforcement_param
        self.countries = countries
        self.colors = colors
        self.layout = layout
        self.label_options = label_options
        self.world = World(self.map_graph, self.presence_map, self.players, self.reinforcement_param)

        self.agents = {i: PolicyGradientAgent() for i in range(self.players)}

    def reset_world(self):
        """Resets the world to its initial state"""
        self.world = World(self.map_graph, self.presence_map, self.players)

    def resolve_battle(self, player1, player2, t_orig, t_dest, N=50):
        """Updates the presence map according to the battle outcome"""
        troop1 = min(self.world.presence_map[player1][t_orig], N)
        troop2 = min(self.world.presence_map[player2][t_dest], N)
        proba = self.proba_table[troop1, troop2]
        rng = np.random.random()

        if rng <= proba:
            self.world.presence_map[player2, t_dest] = 0
            n = troop1 - 1
            self.world.presence_map[player1, t_dest] = n
            self.world.presence_map[player1, t_orig] -= n
        else:
            self.world.presence_map[player1, t_orig] = 1

        return proba

    def turn(self):
        """Runs a turn of the game"""
        for p in range(self.players):

            # Check if player is still in the game
            if len(self.world.get_territories(p)) > 0:

                # Mobilization phase
                player_presence_map = torch.tensor(self.world.get_player_presence_map(p)).float()
                reinforcements = torch.tensor([self.world.get_reinforcements(p)])
                t = self.agents[p].choose_deploy(reinforcements, player_presence_map)
                self.world.deploy(p, t, reinforcements)

                # Attack phase
                player_presence_map = torch.tensor(self.world.get_player_presence_map(p)).float()
                attacks = self.world.get_available_targets(p)

                if len(attacks) > 0:
                    t_orig, t_dest = self.agents[p].choose_attack(attacks, player_presence_map)
                    self.resolve_battle(p, self.world.get_owner(t_dest),
                                        t_orig, t_dest)

                # Fortification phase
                player_presence_map = torch.tensor(self.world.get_player_presence_map(p)).float()
                fortifications = self.world.get_available_fortifications(p)

                if len(fortifications) > 0:
                    t_orig, t_dest = self.agents[p].choose_fortify(fortifications, player_presence_map)
                    self.world.fortify(p, t_orig, t_dest)

    def visualize(self):
        """Visualizes the game with networkx package"""
        draw_map(self.world.map_graph, self.world.presence_map,
                 countries=self.countries, layout=self.layout,
                 label_options=self.label_options)

    def visualize_and_save(self, title=None, save_path=None):
        """Visualizes the game with networkx package and saves it"""
        draw_map_and_save(self.world.map_graph, self.world.presence_map,
                          title=title, filename=save_path,
                          countries=self.countries, layout=self.layout,
                          label_options=self.label_options)

    def run(self, max_turns=20):
        """Runs the game until it is over"""
        self.reset_world()
        self.cur_turn = 0
        self.game_over = False

        while not self.game_over and self.cur_turn < max_turns:
            self.cur_turn += 1
            self.turn()
            # Check if the game is over
            self.game_over = self.world.check_game_over()

    def run_and_save(self, max_turns=20):
        """Runs the game until it is over and saves the game states"""
        self.reset_world()
        self.cur_turn = 0
        self.game_over = False

        while not self.game_over and self.cur_turn < max_turns:
            # Add a 0 to the turn number if it is less than 10
            cur_turn_str = str(self.cur_turn) if self.cur_turn >= 10 else "0" + str(self.cur_turn)

            self.visualize_and_save(title="Turn {}".format(cur_turn_str),
                                    save_path="images/turn_{}.png".format(cur_turn_str))
            self.turn()
            self.cur_turn += 1

            # Check if the game is over
            self.game_over = self.world.check_game_over()

        self.visualize_and_save(title="Turn {} (last turn)".format(cur_turn_str),
                                save_path="images/turn_{}.png".format(cur_turn_str))

    def run_REINFORCE(self, max_turns=1000):
        """Runs the game and apply the REINFORCE algorithm
           to the agent 0"""
        self.reset_world()
        self.cur_turn = 0
        self.game_over = False

        deploy_log_probs = []
        attack_log_probs = []
        fortify_log_probs = []

        deploy_rewards = []
        attack_rewards = []
        fortify_rewards = []

        while not self.game_over and self.cur_turn < max_turns:
            self.cur_turn += 1

            for p in range(self.players):

                # Interrupt the game if the first player is out of the game
                if len(self.world.get_territories(p)) == 0:
                    break

                # Check if player is still in the game
                if len(self.world.get_territories(p)) > 0:

                    # Learning agent
                    if p == 0:
                        # Mobilization phase
                        player_presence_map = torch.tensor(self.world.get_player_presence_map(p)).float()
                        reinforcements = torch.tensor([self.world.get_reinforcements(p)])
                        t, deploy_log_prob = self.agents[p].choose_deploy_prob(reinforcements,
                                                                               player_presence_map)

                        # Make a copy of the world and deploy
                        world_copy = copy.deepcopy(self.world)
                        self.world.deploy(p, t, reinforcements)

                        # Get deploy reward
                        deploy_reward = self.world.get_deploy_reward(world_copy, p)

                        # Save rewards and log probs
                        deploy_rewards.append(deploy_reward)
                        deploy_log_probs.append(deploy_log_prob)

                        # Attack phase
                        player_presence_map = torch.tensor(self.world.get_player_presence_map(p)).float()
                        available_attacks = self.world.get_available_targets(p)
                        attack_log_prob = torch.tensor([0]).float()

                        if len(available_attacks) > 0:
                            t_orig, t_dest, attack_log_prob = self.agents[p].choose_attack_prob(available_attacks,
                                                                                                player_presence_map)

                            # Make a copy of the world and attack
                            world_copy = copy.deepcopy(self.world)
                            win_proba = self.resolve_battle(p, self.world.get_owner(t_dest),
                                                            t_orig, t_dest)

                        # Get attack reward
                        attack_reward = self.world.get_attack_reward(world_copy, p, win_proba)

                        # Save rewards and log probs
                        attack_rewards.append(attack_reward)
                        attack_log_probs.append(attack_log_prob)

                        # Fortification phase
                        player_presence_map = torch.tensor(self.world.get_player_presence_map(p)).float()
                        fortifications = self.world.get_available_fortifications(p)
                        fortify_log_prob = torch.tensor([0]).float()

                        if len(fortifications) > 0:
                            t_orig, t_dest, fortify_log_prob = self.agents[p].choose_fortify_prob(fortifications,
                                                                                                  player_presence_map)
                            # Make a copy of the world and fortify
                            world_copy = copy.deepcopy(self.world)
                            self.world.fortify(p, t_orig, t_dest)

                        # Get fortify reward
                        fortify_reward = self.world.get_fortify_reward(world_copy, p)

                        # Save rewards and log probs
                        fortify_rewards.append(fortify_reward)
                        fortify_log_probs.append(fortify_log_prob)

                    # Other agents
                    else:
                        # Mobilization phase
                        player_presence_map = torch.tensor(self.world.get_player_presence_map(p)).float()
                        reinforcements = torch.tensor([self.world.get_reinforcements(p)])
                        t = self.agents[p].choose_deploy(reinforcements, player_presence_map)
                        self.world.deploy(p, t, reinforcements)

                        # Attack phase
                        player_presence_map = torch.tensor(self.world.get_player_presence_map(p)).float()
                        attacks = self.world.get_available_targets(p)

                        if len(attacks) > 0:
                            t_orig, t_dest = self.agents[p].choose_attack(attacks, player_presence_map)
                            self.resolve_battle(p, self.world.get_owner(t_dest),
                                                t_orig, t_dest)

                        # Fortification phase
                        player_presence_map = torch.tensor(self.world.get_player_presence_map(p)).float()
                        fortifications = self.world.get_available_fortifications(p)

                        if len(fortifications) > 0:
                            t_orig, t_dest = self.agents[p].choose_fortify(fortifications, player_presence_map)
                            self.world.fortify(p, t_orig, t_dest)

                # Check if the game is over
                self.game_over = self.world.check_game_over()

        # Return the monitored values
        return (deploy_rewards, attack_rewards, fortify_rewards,
                deploy_log_probs, attack_log_probs, fortify_log_probs)

    def train_REINFORCE(self, num_games=100, max_turns=100, disp_tqdm=True):
        """Trains the agent using the REINFORCE algorithm
           for a given number of games and a maximum number
           of turns per game"""

        if num_games == 0:
            return None

        deploy_rewards_deque = deque(maxlen=num_games)
        attack_rewards_deque = deque(maxlen=num_games)
        fortify_rewards_deque = deque(maxlen=num_games)

        # Iterate over the episodes
        if disp_tqdm:
            iterator = tqdm(range(1, num_games + 1), desc="Training")
        else:
            iterator = range(1, num_games + 1)

        for game in iterator:
            (deploy_rewards, attack_rewards, fortify_rewards,
             deploy_log_probs, attack_log_probs, fortify_log_probs) = self.run_REINFORCE(max_turns=max_turns)

            if self.cur_turn > 0:
                # Save the score
                deploy_rewards_deque.append(sum(deploy_rewards))
                attack_rewards_deque.append(sum(attack_rewards))
                fortify_rewards_deque.append(sum(fortify_rewards))

                # Calculate the return
                deploy_returns = deque(maxlen=max_turns)
                attack_returns = deque(maxlen=max_turns)
                fortify_returns = deque(maxlen=max_turns)
                n_steps = len(deploy_rewards)
                gamma = self.agents[0].gamma

                for t in range(n_steps)[::-1]:
                    deploy_disc_return_t = deploy_returns[0] if len(deploy_returns) > 0 else 0
                    deploy_returns.appendleft(gamma * deploy_disc_return_t + deploy_rewards[t])

                    attack_disc_return_t = attack_returns[0] if len(attack_returns) > 0 else 0
                    attack_returns.appendleft(gamma * attack_disc_return_t + attack_rewards[t])

                    fortify_disc_return_t = fortify_returns[0] if len(fortify_returns) > 0 else 0
                    fortify_returns.appendleft(gamma * fortify_disc_return_t + fortify_rewards[t])

                # standardization of the returns is employed to make training more stable
                eps = np.finfo(np.float32).eps.item()

                deploy_returns = torch.tensor(deploy_returns)
                deploy_returns = (deploy_returns - deploy_returns.mean()) / (deploy_returns.std() + eps)

                attack_returns = torch.tensor(attack_returns)
                attack_returns = (attack_returns - attack_returns.mean()) / (attack_returns.std() + eps)

                fortify_returns = torch.tensor(fortify_returns)
                fortify_returns = (fortify_returns - fortify_returns.mean()) / (fortify_returns.std() + eps)

                # Compute the deploy loss
                deploy_policy_loss = []
                for log_prob, disc_return in zip(deploy_log_probs, deploy_returns):
                    deploy_policy_loss.append(-log_prob * disc_return)
                deploy_policy_loss = torch.cat(deploy_policy_loss).sum()

                # Compute the attack loss
                attack_policy_loss = []
                for log_prob, disc_return in zip(attack_log_probs, attack_returns):
                    attack_policy_loss.append(-log_prob * disc_return)
                attack_policy_loss = torch.cat(attack_policy_loss).sum()

                # Compute the fortify loss
                fortify_policy_loss = []
                for log_prob, disc_return in zip(fortify_log_probs, fortify_returns):
                    fortify_policy_loss.append(-log_prob * disc_return)
                fortify_policy_loss = torch.cat(fortify_policy_loss).sum()

                # Gradient descent only if the loss is not nan and there are gradients
                if (not torch.isnan(deploy_policy_loss)
                        and not torch.isnan(attack_policy_loss)
                        and not torch.isnan(fortify_policy_loss)):

                    if deploy_policy_loss.grad is not None:
                        self.agents[0].deploy_optimizer.zero_grad()
                        deploy_policy_loss.backward()
                        self.agents[0].deploy_optimizer.step()

                    if attack_policy_loss.grad is not None:
                        self.agents[0].attack_optimizer.zero_grad()
                        attack_policy_loss.backward()
                        self.agents[0].attack_optimizer.step()

                    if fortify_policy_loss.grad is not None:
                        self.agents[0].fortify_optimizer.zero_grad()
                        fortify_policy_loss.backward()
                        self.agents[0].fortify_optimizer.step()

    def eval_REINFORCE(self, num_games=10, max_turns=100, player=0, disp_tqdm=True):
        """Evaluates the agent using the REINFORCE algorithm
           for a given number of games and a maximum number
           of turns per game"""

        wins = 0
        defeats = 0
        ties = 0
        nb_turns = []

        # Iterate over the episodes
        if disp_tqdm:
            iterator = tqdm(range(1, num_games + 1), desc="Evaluation")
        else:
            iterator = range(1, num_games + 1)

        for game in iterator:
            self.run(max_turns=max_turns)

            # Add a win if player won
            if len(self.world.get_territories(player)) == self.presence_map.shape[1]:
                wins += 1
            elif len(self.world.get_territories(player)) == 0:
                defeats += 1
            else:
                ties += 1

            # Add the number of turns
            nb_turns.append(self.cur_turn)

        win_rate = wins / num_games
        defeat_rate = defeats / num_games
        tie_rate = ties / num_games

        return (win_rate, defeat_rate, tie_rate, nb_turns)
