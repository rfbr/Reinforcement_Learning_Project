import numpy as np
import random
from copy import deepcopy
import pickle


class EpsGreedyAgent:
    def __init__(self, name, epsilon):
        self.name = name
        self.epsilon = epsilon
        self.states = []
        self.learning_rate = 0.2
        self.decay_gamma = 0.9
        self.board_values = {}

    def action(self, board):
        positions = board.get_available_positions()
        if np.random.rand() <= self.epsilon:
            action = random.choice(positions)
        else:
            max_value = np.NINF
            action = None
            for position in positions:
                next_move = deepcopy(board)
                next_move.board[position] = self.name
                key = next_move.board_to_text()
                if self.board_values.get(key):
                    action_value = self.board_values.get(key)
                else:
                    action_value = 0
                if action_value > max_value:
                    max_value = action_value
                    action = position
        return action

    def reward(self, reward):
        for board in reversed(self.states):
            if not self.board_values.get(board):
                self.board_values[board] = 0
            self.board_values[board] += self.learning_rate * (
                self.decay_gamma * reward - self.board_values[board])
            reward = self.board_values[board]

    def clear_states(self):
        self.states = []

    def save_policy(self, file):
        fw = open(file, 'wb')
        pickle.dump(self.board_values, fw)
        fw.close()

    def load_policy(self, file):
        fr = open(file, 'rb')
        self.board_values = pickle.load(fr)
        fr.close()
