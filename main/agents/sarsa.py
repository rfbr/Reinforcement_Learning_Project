"""
Implements the SARSA
"""
from main.agents.value_agent import ValueAgent


class SARSA(ValueAgent):
    def __init__(self, name, epsilon):
        ValueAgent.__init__(self, name, epsilon)
        self.display_name = 'SARSA, epsilon = ' + str(epsilon)

    def update_Q(self, state, action, state_, action_, reward):
        if not state:
            return
        q = self.Q[state][action]
        if not state_:
            q += self.alpha * (reward - q)
        else:
            q_ = self.Q[state_][action_]
            q += self.alpha * (reward + self.decay_gamma * q_ - q)
        self.Q[state][action] = q
