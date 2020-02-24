import os
from main.agents import eps_greedy
from main.tic_tac_toe import TicTacToe

if __name__ == '__main__':
    os.system('clear')
    while True:
        try:
            value = int(input(
                '''Welcome to the tic-tac-toe RL game! Choose your agent:
                \t - 0 to train the epsilon-greedy algorithm (epsilon-greedy player VS epsilon-greedy player)
                \n'''
            ))
            if value == 0:
                os.system('clear')
                print("Training the epsilon greedy agent!")
                player_1 = eps_greedy.EpsGreedyAgent(name=1, epsilon=0.3)
                player_2 = eps_greedy.EpsGreedyAgent(name=-1, epsilon=0.3)
                environment = TicTacToe(player_1, player_2)
                environment.play(1)
                break
            else:
                raise ValueError
        except ValueError:
            print("Invalid input :'( Try again")
            pass
