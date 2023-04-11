# import numpy as np


class PolicyGradientAgent():
    def __init__(self):
        self.Q_deploy = None
        self.Q_attack = None
        self.Q_fortify = None

    def choose_deploy(self, reinforcements, world):
        """Returns the territories to deploy troops on"""
        pass

    def choose_attack(self, attack_outcomes, world):
        """Returns the territory to attack"""
        pass

    def choose_conquest(self, world):
        """Returns the number of troops transfered
           to the conquered territory"""
        pass

    def choose_fortify(self, world):
        """Returns the territory to fortify at the end of the turn"""
        pass


class Human_player():
    def __init__(self):
        pass

    def choose_deploy(self):
        pass

    def choose_attack(self):
        pass

    def choose_fortify(self):
        pass
