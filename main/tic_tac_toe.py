import numpy as np
from tqdm import tqdm


class TicTacToe:
    def __init__(self, player_1, player_2):
        self.board = np.zeros((3, 3))
        self.player_1 = player_1
        self.player_2 = player_2

    def get_available_positions(self):
        available_positions = np.where(self.board == 0)
        return list(zip(available_positions[0], available_positions[1]))

    def update_board(self, player, position):
        self.board[position] = player

    def check_win(self, player, position):
        # Check column win
        abs_sum = abs(sum([self.board[row, position[1]] for row in range(3)]))
        if abs_sum == 3:
            return player
        # Check row win
        abs_sum = abs(sum([self.board[position[0], col] for col in range(3)]))
        if abs_sum == 3:
            return player
        # Check diagonal win if last played position was in a diagonal
        if position[0] == position[1]:
            abs_sum = abs(sum([self.board[i, i] for i in range(3)]))
            if abs_sum == 3:
                return player
        if position[0] == (2 - position[1]) or position[1] == (2 - position[0]):
            abs_sum = abs(sum([self.board[i, 2 - i] for i in range(3)]))
            if abs_sum == 3:
                return player
        # Check draw
        if len(self.get_available_positions()) == 0:
            return 0
        return 2

    def board_to_text(self):
        return str(self.board.flatten())

    def give_reward(self, result):
        if result == 1:
            self.player_1.reward(1)
            self.player_2.reward(0)
        elif result == -1:
            self.player_1.reward(0)
            self.player_2.reward(1)
        else:
            self.player_1.reward(.5)
            self.player_2.reward(.5)

    def clear_board(self):
        self.board = np.zeros((3, 3))

    def display_board(self):
        for i in range(0, 3):
            print('-------------')
            out = '| '
            for j in range(0, 3):
                if self.board[i, j] == 1:
                    token = 'x'
                elif self.board[i, j] == -1:
                    token = 'o'
                else:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')

    def display_results(self, win, tie, loss):
        name_1 = self.player_1.__class__.__name__ + ', eps=' + \
            str(self.player_1.epsilon) if self.player_1.__class__.__name__ == 'EpsGreedyAgent' \
            else self.player_1.__class__.__name__
        name_2 = self.player_2.__class__.__name__ + ', eps=' + \
            str(self.player_2.epsilon) if self.player_2.__class__.__name__ == 'EpsGreedyAgent' \
            else self.player_2.__class__.__name__

        str_1 = f'|Player 1: {name_1}' + ' '*(30 - len(name_1)) + '|' + f'{win}' + ' '*(10-len(str(win))) + '|' + \
                f'{tie}' + ' '*(10-len(str(tie))) + '|' + f'{loss}' + ' '*(10-len(str(loss))) + '|'
        str_2 = f'|Player 2: {name_2}' + ' '*(30 - len(name_2)) + '|' + f'{loss}' + ' '*(10-len(str(loss))) + '|' + \
                f'{tie}' + ' '*(10-len(str(tie))) + '|' + f'{win}' + ' '*(10-len(str(win))) + '|'
        n = max(len(str_1), len(str_2))
        print('-'*n)
        print('|Agent    ' + ' '*30 + ' |Win       |Draw      |Loss      |')
        print('-'*n)
        print(str_1)
        print('-'*n)
        print(str_2)
        print('-'*n)

    def play(self, player):
        action = player.action(self)
        self.update_board(player.name, action)
        player.states.append(self.board_to_text())
        win_or_draw = self.check_win(player.name, action)
        # self.display_board()
        if abs(win_or_draw) in (0, 1):
            self.give_reward(win_or_draw)
            self.player_1.clear_states()
            self.player_2.clear_states()
            self.clear_board()
            return win_or_draw
        return None

    def train(self, nb_games):
        players = {
            0: self.player_1,
            1: self.player_2
        }
        for _ in tqdm(range(nb_games)):
            first_play = True
            i = 0
            while True:
                if first_play:
                    player = int(np.random.rand() < .5)
                    first_play = False
                else:
                    i += 1
                ind = (i + player) % 2
                result = self.play(players[ind])
                if result in [-1, 0, 1]:
                    break

    def simulation(self, nb_games):
        win, draw, loss = 0, 0, 0
        players = {
            0: self.player_1,
            1: self.player_2
        }
        for _ in tqdm(range(nb_games)):
            first_play = True
            i = 0
            while True:
                if first_play:
                    player = int(np.random.rand() < .5)
                    first_play = False
                else:
                    i += 1
                result = self.play(players[(i + player) % 2])
                if result == 1:
                    win += 1
                    break
                elif result == 0:
                    draw += 1
                    break
                elif result == -1:
                    loss += 1
                    break
        self.display_results(win, draw, loss)
