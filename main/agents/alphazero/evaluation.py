import os
import torch
import numpy as np
from tqdm import tqdm
from main.env.tic_tac_toe import TicTacToe
from main.agents.alphazero.net import Net
from main.agents.alphazero.mcts import mcts_simulation, compute_policy


def evaluate_nets(iteration_1, iteration_2):
    current_net = "%s_iter%d.pt" % ('AlphaZero', iteration_2)
    best_net = "%s_iter%d.pt" % ('AlphaZero', iteration_1)
    current_net_filename = os.path.join(
        "./main/agents/alphazero/data/model_data/", current_net)
    best_net_filename = os.path.join(
        "./main/agents/alphazero/data/model_data/", best_net)

    current_cnet = Net()
    best_cnet = Net()
    if torch.cuda.is_available():
        current_cnet.cuda()
        best_cnet.cuda()

    current_cnet.eval()
    best_cnet.eval()
    checkpoint = torch.load(current_net_filename)
    current_cnet.load_state_dict(checkpoint['state_dict'])
    checkpoint = torch.load(best_net_filename)
    best_cnet.load_state_dict(checkpoint['state_dict'])
    tictactoe = EvaluationEnv(current_cnet, best_cnet)
    win_ration = tictactoe.evaluate(10)

    if win_ration >= .55:
        return iteration_2
    else:
        return iteration_1


class EvaluationEnv(TicTacToe):
    def __init__(self, current_net, saved_net):
        TicTacToe.__init__(self, current_net, saved_net)

    def play_round(self):
        endgame = False
        value = 0
        if np.random.rand() < .5:
            one_player = self.player_1
            minus_one_player = self.player_2
            current = 1
            player = 1
        else:
            one_player = self.player_2
            minus_one_player = self.player_1
            current = -1
            player = -1

        while endgame == False:

            if player == 1:
                root = mcts_simulation(self, 200, one_player, player)
                policy = compute_policy(root)
            else:
                root = mcts_simulation(self, 200, minus_one_player, player)
                policy = compute_policy(root)
            move = np.random.choice(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
                                    1,
                                    p=policy)
            move = (move // 3, move % 3)
            self.update_board(player, move)
            win_or_draw = self.check_win(player, move)
            player *= -1
            if abs(win_or_draw) in (0, 1):
                value = win_or_draw * current  #1 if current net wins
                self.clear_board()
                endgame = True
        return value

    def evaluate(self, num_games):
        current_wins = 0
        for _ in tqdm(range(num_games)):
            with torch.no_grad():
                winner = self.play_round()
            if winner == 1:
                current_wins += 1
        win_ration = current_wins / num_games
        print(f"Current_net wins ratio: {win_ration}")
        return win_ration