"""
Generic Value Based agent
"""
from numpy import random


class ValueAgent:
    def __init__(self, name, epsilon):
        self.name = name
        self.epsilon = epsilon
        self.states = []
        self.learning_rate = 0.2
        self.decay_gamma = 0.9
        self.alpha = 0.8
        self.old_state = None
        self.old_action = None
        self.new_state = None
        self.new_action = None
        self.Q = {}

    def update_Q(self, state, action, state_, action_, reward):
        raise NotImplemented()

    def action(self, board):
        positions = board.get_available_positions()
        # epsilon-greedy
        state = board.board_to_text()
        if not self.Q.get(state):
            self.Q[state] = {}
            for position in positions:
                self.Q[state][position] = 0
        # Explore
        if random.random() < self.epsilon:
            action = positions[random.randint(len(positions))]
        # Exploit
        else:
            action = max(positions, key=lambda x: self.Q[state][x])
        # Update actions
        if len(positions) == 9:
            self.update_Q(self.old_state, self.old_action, None, None, 0)
            self.old_state = state
            self.old_action = action
        else:
            if len(positions) > 1:
                self.update_Q(self.old_state, self.old_action, state, action, 0)
                self.old_state = state
                self.old_action = action
        self.new_action = action
        self.new_state = state
        return action

    def reward(self, reward):
        self.update_Q(self.old_state, self.old_action, self.new_state, self.new_action, reward)
        self.old_state = self.new_state
        self.old_action = self.new_action

    def clear_states(self):
        self.states = []
        self.old_state = None
        self.old_action = None
        self.new_state = None
        self.new_action = None

    def save_policy(self, file):
        pass

    def load_policy(self, file):
        pass
