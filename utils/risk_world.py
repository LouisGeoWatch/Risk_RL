import numpy as np

# World Map
# 0 : North America
# 1 : South America
# 2 : Europe
# 3 : Africa
# 4 : Asia
# 5 : South-Asia / Oceania
map_graph = np.array([
    [0,1,1,0,1,0],
    [1,0,1,0,0,0],
    [1,1,0,1,1,0],
    [0,1,1,0,0,0],
    [1,0,1,0,0,1],
    [0,0,0,0,1,0]
])

# Nb player
nb_player = 3

# Vector of presence on the map
possession_map = np.random.choice(nb_player, len(map_graph))

class Game():
    def __init__(self):
        self.map = map_graph
        self.player_map = possession_map
        self.players = nb_player
        self.turn = 0
        self.phase = 0
    
    def next_player_turn(self):
        self.turn = (self.turn + 1) % self.players
    
    def next_phase(self):
        self.phase = (self.phase + 1) % 3

    def update_after_conquest(self, winner_name, siege):
        self.player_map[siege] = winner_name
    