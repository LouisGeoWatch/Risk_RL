import numpy as np

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
nb_players = 3

# Vector of presence on the map
# P[i, j] = #troops of player i in zone j
presence_map = np.array([
    [2, 2, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 2]
])


class World():
    def __init__(self, map_graph, presence_map, nb_players):
        self.map_graph = map_graph
        self.presence_map = presence_map
        self.nb_players = nb_players

    def get_territories(self, p):
        """Returns the territories belonging to player p
            as a list of vertices"""
        return [i for i in range(self.map_graph.shape[0])
                if self.presence_map[p][i]]

    def deploy(self, p, t, n):
        """Deploys n troops for player p on territory t"""
        self.presence_map[p][t] += n

    def retreat(self, n, p, t_orig, t_dest):
        """Relocates n troops of player p
            from territory t_orig to territory t_dest"""
        self.presence_map[p, t_orig] -= n
        self.presence_map[p, t_dest] += n

    def get_owner(self, t):
        """Returns the player p owning the territory t"""
        return np.argmax(self.presence_map[:, t])

    def get_neighbors(self, t):
        """Returns all the neighboring territories of t
            as a list of vertices"""
        return [i for i in range(self.map_graph.shape[0])
                if self.map_graph[t][i]]

    def get_t_neighbors_pairs(self, t):
        """Returns all the neighboring territories of t
            as a list of edges"""
        return [(t, a) for a in range(self.map_graph.shape[0])
                if self.map_graph[t][a]]

    def get_available_targets(self, p):
        """Returns the possible attack targets of player p
            as a list of vertices"""
        t_list = self.get_territories(p)
        target_list = []

        for t in t_list:
            target_list += self.get_neighbors(t)

        return list(set(target_list))

    def get_available_target_pairs(self, p):
        """Returns the possible attack targets of player p
            as a list of edges"""
        t_list = self.get_territories(p)
        target_list = []

        for t in t_list:
            target_list += self.get_t_neighbors_pairs(t)

        return target_list

    def get_reinforcements(self, p):
        """Returns the number of reinforcements
            based on the number ot territories
            controlled by player p"""
        return 3*len(self.get_territories(p))
