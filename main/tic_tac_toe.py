import numpy as np
from tqdm import tqdm


class TicTacToe:
    def __init__(self, player_1, player_2, nb_games):
        self.board = np.zeros((3, 3))
        self.endgame = False
        self.player_1 = player_1
        self.player_2 = player_2
        self.nb_games = nb_games

    def get_available_positions(self):
        available_positions = np.where(self.board == 0)
        return list(zip(available_positions[0], available_positions[1]))

    def update_board(self, player, position):
        self.board[position] = player

    def check_win(self, player, position):
        # Check column win
        abs_sum = abs(sum([self.board[row, position[1]] for row in range(3)]))
        if abs_sum == 3:
            self.endgame = True
            return player
        # Check row win
        abs_sum = abs(sum([self.board[position[0], col] for col in range(3)]))
        if abs_sum == 3:
            self.endgame = True
            return player
        # Check diagonal win if last played position was in a diagonal
        if position[0] == position[1]:
            abs_sum = abs(sum([self.board[i, i] for i in range(3)]))
            if abs_sum == 3:
                self.endgame = True
                return player
        if position[0] == (2 - position[1]) or position[1] == (2 - position[0]):
            abs_sum = abs(sum([self.board[i, 2 - i] for i in range(3)]))
            if abs_sum == 3:
                self.endgame = True
                return player
        # Check draw
        if len(self.get_available_positions()) == 0:
            self.endgame = True
            return 0

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
        self.endgame = False

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

    def display_results(self, win, tie, loose):
        print('-----------------------------------------')
        print(
            f'|Player 1: {self.player_1.__class__.__name__} | {win} | {tie} | {loose}|')
        print(
            f'|Player 2: {self.player_2.__class__.__name__} | {loose} | {tie} | {win}|')
        print('-----------------------------------------')

    def play(self):
        win, tie, loose = 0, 0, 0
        for _ in tqdm(range(self.nb_games)):
            player_1_win_or_draw = None
            player_2_win_or_draw = None
            while not self.endgame:
                # Player 1 plays
                player_1 = self.player_1.name
                player_1_action = self.player_1.action(self)
                self.update_board(player_1, player_1_action)
                # Add board configuration to player's states
                self.player_1.states.append(self.board_to_text())
                # Check if player 1 won with this move
                player_1_win_or_draw = self.check_win(
                    player_1, player_1_action)

                if player_1_win_or_draw in (0, 1):
                    self.give_reward(player_1_win_or_draw)
                    self.player_1.clear_states()
                    self.player_2.clear_states()
                    self.clear_board()
                    break
                else:
                    # Player 2 plays
                    player_2 = self.player_2.name
                    player_2_action = self.player_2.action(self)
                    self.update_board(player_2, player_2_action)
                    # Add board configuration to player's states
                    self.player_2.states.append(self.board_to_text())
                    # Check if player 2 won with this move
                    player_2_win_or_draw = self.check_win(
                        player_2, player_2_action)

                    if player_2_win_or_draw in (0, -1):
                        self.give_reward(player_2_win_or_draw)
                        self.player_2.clear_states()
                        self.player_2.clear_states()
                        self.clear_board()
                        break

            if player_1_win_or_draw == 0 or player_2_win_or_draw == 0:
                tie += 1
            elif player_1_win_or_draw:
                win += 1
            else:
                loose += 1

        self.display_results(win, tie, loose)
