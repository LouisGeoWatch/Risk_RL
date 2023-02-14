import numpy as np

# World Map
# 0 : North America
# 1 : South America
# 2 : Europe
# 3 : Africa
# 4 : Asia
# 5 : South-Asia / Oceania
graph = np.array([
    [0,1,1,0,1,0],
    [1,0,1,0,0,0],
    [1,1,0,1,1,0],
    [0,1,1,0,0,0],
    [1,0,1,0,0,1],
    [0,0,0,0,1,0]
])

# Nb player
nb_player = 3

class Environnement():
    def __init__(self):
        self.map = graph
        self.players = nb_player
        self.turn = 0
        self.phase = 0
    
    def next_player_turn(self):
        self.turn = (self.turn + 1) % self.players
    
    def next_phase(self):
        self.phase = (self.phase + 1) % 3
    
    