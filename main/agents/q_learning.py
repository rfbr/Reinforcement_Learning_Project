"""
Implements the Q-Learning
"""
from main.agents import value_agent


class QLearning(value_agent.ValueAgent):
    def update_Q(self, state, action, state_, action_, reward):
        if not state:
            return
        q = self.Q[state][action]
        if not state_:
            q += self.alpha * (reward - q)
        else:
            q_ = max(self.Q[state_].values())
            q += self.alpha * (reward + self.decay_gamma * q_ - q)
        self.Q[state][action] = q
