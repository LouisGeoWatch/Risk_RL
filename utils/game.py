import numpy as np
import copy
from utils.world import World
from utils.agent import PolicyGradientAgent
from utils.viz import draw_graph

# World Map
map_graph = np.array([
    [0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0],
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

# Probabilities of winning
proba_table = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [42, 11, 3, 1, 0, 0, 0, 0],
    [75, 36, 20, 9, 5, 2, 1, 0],
    [92, 65, 47, 31, 21, 14, 8, 5],
    [97, 78, 64, 47, 36, 25, 18, 12],
    [99, 89, 77, 64, 50, 40, 29, 22],
    [100, 93, 86, 75, 64, 52, 42, 33],
    [100, 97, 91, 83, 74, 64, 53, 45]
])

colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow',
          4: 'purple', 5: 'orange', 6: 'black', 7: 'white'}


class Game():
    def __init__(self, map_graph=map_graph, presence_map=presence_map,
                 nb_players=nb_player, proba_table=proba_table,
                 countries=countries, colors=colors):
        self.map_graph = map_graph.copy()
        self.presence_map = presence_map.copy()
        self.players = nb_players
        self.proba_table = proba_table
        self.cur_turn = 0
        self.phase = 0
        self.game_over = False
        self.countries = countries
        self.colors = colors

        self.world = World(self.map_graph, self.presence_map, self.players)
        self.agents = {i: PolicyGradientAgent() for i in range(self.players)}

    def reset_world(self):
        """Resets the world to its initial state"""
        self.world = World(self.map_graph, self.presence_map, self.players)

    def resolve_battle(self, player1, player2, t_orig, t_dest):
        """Updates the presence map according to the battle outcome"""
        troop1 = min(self.world.presence_map[player1][t_orig], 7)
        troop2 = min(self.world.presence_map[player2][t_dest], 7)
        proba = self.proba_table[troop1, troop2]/100
        rng = np.random.random()

        if rng <= proba:
            print("Battle won!")
            self.world.presence_map[player2, t_dest] = 0
            n = troop1 - 1
            self.world.presence_map[player1, t_dest] = n
            self.world.presence_map[player1, t_orig] -= n
        else:
            print("Battle lost!")
            self.world.presence_map[player1, t_orig] = 1

    def turn(self):
        """Runs a turn of the game"""
        for p in range(self.players):

            # Mobilization phase
            reinforcements = self.world.get_reinforcements(p)
            t, n = self.agents[p].choose_deploy(reinforcements, self.world)
            self.world.deploy(p, t, n)

            # Attack phase
            available_attacks = self.world.get_available_targets(p)
            attack = self.agents[p].choose_attack(available_attacks, self.world)
            if attack is not None:
                t_orig, t_dest = attack
                self.resolve_battle(p, self.world.get_owner(t_dest),
                                    t_orig, t_dest)

            # Fortification phase
            t_orig, t_dest, n = self.agents[p].choose_fortify(self.world)
            self.world.fortify(p, t_orig, t_dest, n)

        # Check if game is over
        player_troops = np.sum(self.world.presence_map, axis=1)
        player_remaining = np.count_nonzero(player_troops)
        if player_remaining == 1:
            self.game_over = True

    def run(self):
        """Runs the game until it is over"""
        while not self.game_over():
            self.turn()

    def visualize(self):
        """Visualizes the game with networkx package"""
        labels = {i: f"{i}: {self.countries[i]} ({self.world.presence_map[0, i]})"
                  for i in range(self.map_graph.shape[0])}
        owners = np.argmax(self.world.presence_map, axis=0)
        colors = [self.colors[i] for i in owners]

        draw_graph(self.map_graph, colors, labels)

    def run_REINFORCE(self, max_turns=1000):
        """Runs the game and apply the REINFORCE algorithm
           to the agent 0"""
        self.reset_world()
        self.cur_turn = 0
        self.phase = 0
        self.game_over = False

        deploy_log_probs = []
        attack_log_probs = []
        fortify_log_probs = []

        deploy_rewards = []
        attack_rewards = []
        fortify_rewards = []

        while not self.game_over() and self.cur_turn < max_turns:
            for p in range(self.players):

                # Learning agent
                if p == 0:
                    # Mobilization phase
                    player_presence_map = self.world.get_player_presence_map(p)
                    reinforcements = self.world.get_reinforcements(p)
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
                    player_presence_map = self.world.get_player_presence_map(p)
                    available_attacks = self.world.get_available_targets(p)
                    attack_log_prob = 0

                    if len(available_attacks) > 0:
                        t_orig, t_dest, attack_log_prob = self.agents[p].choose_attack_prob(available_attacks,
                                                                                            player_presence_map)

                        # Make a copy of the world and attack
                        world_copy = copy.deepcopy(self.world)
                        self.resolve_battle(p, self.world.get_owner(t_dest),
                                            t_orig, t_dest)

                    # Get attack reward
                    attack_reward = self.world.get_attack_reward(world_copy, p)

                    # Save rewards and log probs
                    attack_rewards.append(attack_reward)
                    attack_log_probs.append(attack_log_prob)

                    # Fortification phase
                    player_presence_map = self.world.get_player_presence_map(p)
                    fortifications = self.world.get_available_fortifications(p)
                    fortify_log_prob = 0

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
                    player_presence_map = self.world.get_player_presence_map(p)
                    reinforcements = self.world.get_reinforcements(p)
                    t = self.agents[p].choose_deploy(reinforcements, player_presence_map)
                    self.world.deploy(p, t, reinforcements)

                    # Attack phase
                    player_presence_map = self.world.get_player_presence_map(p)
                    attacks = self.world.get_available_targets(p)

                    if len(attacks) > 0:
                        t_orig, t_dest = self.agents[p].choose_attack(attacks, player_presence_map)
                        self.resolve_battle(p, self.world.get_owner(t_dest),
                                            t_orig, t_dest)

                    # Fortification phase
                    player_presence_map = self.world.get_player_presence_map(p)
                    fortifications = self.world.get_available_fortifications(p)

                    if len(fortifications) > 0:
                        t_orig, t_dest = self.agents[p].choose_fortify(fortifications, player_presence_map)
                        self.world.fortify(p, t_orig, t_dest)

        # Update the agent at the end of the game
        self.agents[0].update(deploy_rewards, attack_rewards, fortify_rewards,
                              deploy_log_probs, attack_log_probs, fortify_log_probs)
