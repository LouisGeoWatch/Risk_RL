import numpy as np
from utils.world import World
from utils.agent import Agent

# World Map
# 0 : North America
# 1 : South America
# 2 : Europe
# 3 : Africa
# 4 : Asia
# 5 : South-Asia / Oceania
map_graph = np.array([
    [0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0]
])

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


# class Game():
#     def __init__(self):
#         self.map_graph = map_graph
#         self.presence_map = presence_map
#         self.players = nb_player
#         self.proba_table = proba_table
#         self.turn = 0
#         self.phase = 0

#         self.world = World(self.map_graph, self.presence_map, self.players)

#     def next_player_turn(self):
#         self.turn = (self.turn + 1) % self.players

#     def next_phase(self):
#         self.phase = (self.phase + 1) % 3

#     def mobilize(self, n, player, zone):
#         self.presence_map[player, zone] += n

#     def retreat(self, n, player, zone_from, zone_to):
#         self.presence_map[player, zone_from] -= n
#         self.presence_map[player, zone_to] += n

#     def result_battle(self, player1, player2, zone):
#         # Returns winner first, loser second, and siege
#         troop1 = min(self.presence_map[player1][zone], 7)
#         troop2 = min(self.presence_map[player2][zone], 7)
#         proba = self.proba_table[troop1, troop2]/100
#         rng = np.random.random()
#         if rng <= proba:
#             self.presence_map[player2, zone] = 0
#             return player1, player2, zone
#         else:
#             self.presence_map[player1, zone] = 0
#             return player2, player1, zone

class Game():
    def __init__(self):
        self.map_graph = map_graph
        self.presence_map = presence_map
        self.players = nb_player
        self.proba_table = proba_table
        self.turn = 0
        self.phase = 0
        self.game_over = False

        self.world = World(self.map_graph, self.presence_map, self.players)
        self.agents = {i: Agent() for i in range(self.players)}

    def attack_outcomes(self, p):
        """Returns a list of possible attacks for player p
            and their winning outcome"""
        target_pairs = self.world.get_available_target_pairs(p)

        return [(
                 (t, self.presence_map[p][t]),
                 (a, self.presence_map[self.world.get_owner(a)][a]),
                 proba_table[self.presence_map[p][t]]
                            [self.presence_map[self.world.get_owner(a)][a]]
                 )
                for (t, a) in target_pairs]

    def result_battle(self, player1, player2, zone):
        """Returns winner first, loser second, and siege"""
        troop1 = min(self.presence_map[player1][zone], 7)
        troop2 = min(self.presence_map[player2][zone], 7)
        proba = self.proba_table[troop1, troop2]/100
        rng = np.random.random()
        if rng <= proba:
            self.presence_map[player2, zone] = 0
            return player1, player2, zone
        else:
            self.presence_map[player1, zone] = 0
            return player2, player1, zone

    def turn(self):
        """Runs a turn of the game"""
        for p in range(self.players):

            # Mobilization phase
            reinforcements = self.world.get_reinforcements(p)
            t, n = self.agents[p].choose_deploy(reinforcements)
            self.world.deploy(p, t, n)

            # Attack phase
            attack_outcomes = self.attack_outcomes(p)
            attack = self.agents[p].choose_attack(attack_outcomes)
            if attack is not None:
                t, a = attack
                self.world.attack(p, t, a)
                self.result_battle(p, self.world.get_owner(a), a)

            # Fortification phase
            t_orig, t_dest, n = self.agents[p].choose_fortify()
            self.world.fortify(p, t_orig, t_dest, n)

        # Check if game is over
        pass

    def run(self):
        """Runs the game until it is over"""
        while not self.game_over():
            self.turn()