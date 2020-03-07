"""
Implements the SARSA
"""
import numpy as np
from main.agents.value_agent import ValueAgent


class ExpectedSARSA(ValueAgent):
    def __init__(self, name, epsilon):
        ValueAgent.__init__(self, name, epsilon)
        self.display_name = 'ExpectedSARSA, epsilon = ' + str(epsilon)

    def update_Q(self, state, action, state_, action_, reward):
        if not state:
            return
        q = self.Q[state][action]
        if not state_:
            q += self.alpha * (reward - q)
        else:
            q_ = np.mean(list(self.Q[state_].values()))
            q += self.alpha * (reward + self.decay_gamma * q_ - q)
        self.Q[state][action] = q
