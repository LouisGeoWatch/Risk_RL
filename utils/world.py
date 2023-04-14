import numpy as np


class World():
    def __init__(self, map_graph, presence_map, nb_players):
        self.map_graph = map_graph.copy()
        self.presence_map = presence_map.copy()
        self.nb_players = nb_players

    def get_territories(self, p):
        """Returns the territories belonging to player p
            as a list of vertices"""
        return [i for i in range(self.map_graph.shape[0])
                if self.presence_map[p][i]]
    
    def check_game_over(self):
        check_vector = np.sum(self.presence_map, axis = 1)
        return (np.count_nonzero(check_vector) == 1)

    def deploy(self, p, t, n):
        """Deploys n troops for player p on territory t"""
        self.presence_map[p][t] += n

    def fortify(self, p, t_orig, t_dest):
        """Relocates n troops of player p
            from territory t_orig to territory t_dest"""
        self.presence_map[p, t_dest] += self.presence_map[p, t_orig] - 1
        self.presence_map[p, t_orig] = 1

    def get_owner(self, t):
        """Returns the player p owning the territory t"""
        return np.argmax(self.presence_map[:, t])

    def get_neighbors(self, t):
        """Returns all the neighboring territories of t
            as a list of vertices"""
        return [i for i in range(self.map_graph.shape[0])
                if self.map_graph[t][i]]
    
    def get_neighboring_opponents(self, t, p):
        """Returns the number of p's opponent in the neighboring territories of t
            as a list of vertices
            We suppose that p owns t"""
        opponent_list = []

        available_targets = self.get_available_targets(p)
        neighboring_opponents = [e[1] for e in available_targets if e[0]==t]

        return [np.amax(self.presence_map[:, tau]) for tau in neighboring_opponents]

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
            if self.presence_map[p][t] > 1:
                target_list += [e for e in self.get_t_neighbors_pairs(t) if e[1] not in t_list]

        return list(set(target_list))

    def get_available_fortifications(self, p):
        """Returns the possible fortifications of player p
           (territories with more than 1 troop)"""
        return [(t, a) for t in self.get_territories(p)
                for a in self.get_neighbors(t)
                if self.get_owner(a) == p and self.presence_map[p][t] > 1]

    def get_reinforcements(self, p):
        """Returns the number of reinforcements
            based on the number ot territories
            controlled by player p"""
        return 3*len(self.get_territories(p))

    def get_player_presence_map(self, p):
        """Returns the presence map of player p
            which is its troop being positive
            and the troop of the other players being negative"""
        ennemy_players = [i for i in range(self.nb_players) if i != p]

        return self.presence_map[p] - np.sum(self.presence_map[ennemy_players], axis=0)

    def get_world_evolution(self, previous_world):
        """Returns the evolution of the world
            between the old world and the current world"""
        return self.presence_map - previous_world.presence_map

    def get_world_evolution_player(self, previous_world, p):
        """Returns the evolution of the world
            between the old world and the current world
            for player p"""
        previous_presence_map = previous_world.get_player_presence_map(p)
        new_presence_map = self.get_player_presence_map(p)

        return new_presence_map - previous_presence_map

    def get_deploy_reward(self, previous_world, p):
        """Returns the reward for deploying n troops on territory t using a utility function"""

        world_evolution_p = self.get_world_evolution_player(previous_world, p)
        n, t = np.amax(world_evolution_p), np.argmax(world_evolution_p)

        adv_neighbors_troops = np.array(self.get_neighboring_opponents(t, p))
        nb_adv_territories = len(adv_neighbors_troops)

        barbarian_term = np.sum((adv_neighbors_troops)**2)

        def concave_term(x):
            if x>0:
                return np.log(1+x) - x*np.log(x)
            else:
                return 0
        concave_term = concave_term(n)

        return concave_term + barbarian_term

    def get_attack_reward(self, previous_world, p, win_proba):
        """Returns the reward for attacking territory t"""
        world_evolution_p = self.get_world_evolution_player(previous_world, p)
        troops_losses = np.sum(world_evolution_p)/np.sum(previous_world.presence_map[p, :])

        return win_proba - troops_losses

    def get_fortify_reward(self, previous_world, p):
        """Returns the reward for fortifying territory t"""
        world_evolution_p = self.get_world_evolution_player(previous_world, p)

        #Obtain the number of troops moved and the territories
        f, t_start, t_end = np.amax(world_evolution_p), np.argmax(world_evolution_p), np.argmin(world_evolution_p)

        neighbors_troops_start = self.get_neighboring_opponents(t_start, p)
        neighbors_troops_end = self.get_neighboring_opponents(t_end, p)

        hedge_start = np.min([self.presence_map[p, t_start] - x for x in neighbors_troops_start])
        hedge_end = np.min([self.presence_map[p, t_end] - x for x in neighbors_troops_end])

        delta_hedging = hedge_start+hedge_end

        return (f*delta_hedging if delta_hedging <= 0 else delta_hedging)
