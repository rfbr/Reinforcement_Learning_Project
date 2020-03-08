import numpy as np
from copy import deepcopy
import pickle
import torch
import os
import datetime
from tqdm import tqdm
from main.env.tic_tac_toe import TicTacToe
import main.agents.alphazero.net as alphazero
from main.agents.alphazero.utils import save_data


#Node class for the MCTS
class Node():
    def __init__(self, board, action, player, parent=None, c=1):
        self.board = board
        self.action = action
        self.parent = parent
        self.player = player  #player who will play at this node
        self.c = c  #Degree of exploration
        self.is_a_leaf = True
        self.children = dict()
        self.children_P = np.zeros(9)
        self.children_total_Q = np.zeros(9)
        self.children_N = np.zeros(9)
        self.possible_moves = []

    #Define node.number_of_visits from the parent child number of visists
    @property
    def number_of_visits(self):
        return self.parent.children_N[self.action]

    #Setter associated with node.number_of_visits
    @number_of_visits.setter
    def number_visits(self, value):
        self.parent.children_N[self.action] = value

    #Define node.total_value from the parent child number of visists
    @property
    def total_value(self):
        return self.parent.children_total_Q[self.action]

    # Setter associated with node.total_value
    @total_value.setter
    def total_value(self, value):
        self.parent.children_total_Q[self.action] = value

    #Upper confidence bound
    def children_UCB(self):
        return self.children_total_Q / (
            1 + self.children_N) + self.c * self.children_P * np.sqrt(
                self.number_visits) / (1 + self.children_N)

    def best_move(self):
        if len(self.possible_moves) == 0:
            best_move = np.argmax(self.children_UCB())
        else:
            #Among possible moves, pick the one with the highest UCB
            best_move = self.possible_moves[np.argmax(
                self.children_UCB()[self.possible_moves])]
        return best_move

    def selection(self):
        selected_node = self
        while not selected_node.is_a_leaf:
            best_move = selected_node.best_move()
            if best_move in selected_node.children:
                selected_node = selected_node.children[best_move]
            else:
                #Create a node if the child associated with the best move does not exist
                selected_node = selected_node.add_child(best_move)
        return selected_node

    def expansion(self, children_prior_probability):
        possible_moves = [
            3 * i + j for i, j in self.board.get_available_positions()
        ]
        self.is_a_leaf = True if len(possible_moves) == 0 else False
        self.possible_moves = possible_moves
        for move in range(9):
            if move not in possible_moves:
                children_prior_probability[
                    move] = 0.  # Proba for illegal moves set to 0
        self.children_P = children_prior_probability

    def backpropagation(self, estimated_value):
        selected_node = self
        while selected_node.parent is not None:
            selected_node.number_visits += 1
            if selected_node.player == -1:  #If the parent is 1
                selected_node.total_value += estimated_value
            elif selected_node.player == 1:  #If the parent is -1
                selected_node.total_value += -estimated_value
            selected_node = selected_node.parent

    def add_child(self, action):
        copy_board = deepcopy(self.board)  # make copy of board
        move_2d = (action // 3, action % 3
                   )  #Convert 1d move {1,..,9} to 2d move
        copy_board.update_board(self.player, move_2d)
        new_player = self.player * -1
        self.children[action] = Node(copy_board,
                                     action,
                                     new_player,
                                     parent=self)
        return self.children[action]


class RootValueKeeper(object):
    def __init__(self):
        self.parent = None
        self.children_total_Q = np.zeros(9)
        self.children_N = np.zeros(9)


def mcts_simulation(board, nb_sim, agent, player):
    root_node = Node(board=board,
                     action=0,
                     player=player,
                     parent=RootValueKeeper())
    cuda_availability = torch.cuda.is_available()
    for _ in range(nb_sim):
        selected_leaf = root_node.selection()
        leaf_board = selected_leaf.board.board
        #Add a new channel to the board containing the player information
        net_leaf_board = np.zeros((2, 3, 3))
        net_leaf_board[0] = leaf_board
        net_leaf_board[1] = np.ones(3) * player
        net_leaf_board = torch.from_numpy(net_leaf_board).float()
        if cuda_availability:
            net_leaf_board = net_leaf_board.cuda()
        child_p, child_v = agent(net_leaf_board)

        child_p = child_p.detach().cpu().numpy().reshape(-1)
        child_v = child_v.item()
        move = selected_leaf.action
        if abs(selected_leaf.board.check_win(player,
                                             (move // 3, move % 3))) in (0, 1):
            selected_leaf.backpropagation(child_v)
        else:
            selected_leaf.expansion(child_p)
            selected_leaf.backpropagation(child_v)
    return root_node


def mcts_self_play(agent, num_games, start_index, iteration):
    if not os.path.isdir(
            f"./main/agents/alphazero/data/datasets/iter_{iteration}"):
        if not os.path.isdir("./main/agents/alphazero/data/datasets"):
            os.mkdir("./main/agents/alphazero/data/datasets")
        os.mkdir(f"./main/agents/alphazero/data/datasets/iter_{iteration}")

    for i in tqdm(range(start_index, num_games + start_index)):
        endgame = False
        dataset = []
        value = 0
        move_count = 0
        player = 1 if np.random.rand() < .5 else -1
        board = TicTacToe()
        while endgame == False:
            root = mcts_simulation(board=board,
                                   nb_sim=200,
                                   agent=agent,
                                   player=player)
            policy = compute_policy(root)

            move = np.random.choice(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
                                    1,
                                    p=policy)
            move = (move // 3, move % 3)
            board.update_board(player, move)
            #Add a new channel to the board containing the player information
            net_board = np.zeros((2, 3, 3))
            net_board[0] = deepcopy(board.board)
            net_board[1] = np.ones(3) * player
            dataset.append([net_board, policy])
            win_or_draw = board.check_win(player, move)
            if abs(win_or_draw) in (0, 1):
                if win_or_draw != 0:
                    value = win_or_draw
                else:
                    value = 0.5 * player
                endgame = True
            move_count += 1
            player *= -1

        dataset_p = []
        for idx, data in enumerate(dataset):
            s, p = data
            if idx == 0:
                dataset_p.append([s, p, 0])
            else:
                dataset_p.append([s, p, value])
        del dataset
        save_data(
            dataset_p,
            f"./main/agents/alphazero/data/datasets/iter_{iteration}/{i}_" +
            datetime.datetime.today().strftime("%Y-%m-%d"))


def compute_policy(root, temp=1):
    return (root.children_N)**(1 / temp) / sum(root.children_N**(1 / temp))


def run_mcts(nb_games, start_index=0, iteration=0):
    if not os.path.isdir("./main/agents/alphazero/data/model_data/"):
        os.mkdir("./main/agents/alphazero/data/model_data/")
    net_to_play = "AlphaZero_iter%d.pt" % iteration
    net = alphazero.Net()
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    current_net_filename = "./main/agents/alphazero/data/model_data/" + net_to_play
    if os.path.isfile(current_net_filename):
        checkpoint = torch.load(current_net_filename)
        net.load_state_dict(checkpoint['state_dict'])
        print("Loaded %s model." % current_net_filename)
    else:
        torch.save({'state_dict': net.state_dict()},
                   "./main/agents/alphazero/data/model_data/" + net_to_play)
        print("Initialized model.")
    with torch.no_grad():
        mcts_self_play(net, nb_games, start_index, iteration)