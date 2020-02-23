import os
from main import agents
from main.tictactoe import tictactoe

if __name__ == '__main__':
    os.system('clear')
    while True:
        try:
            value = int(input(
'''Welcome to the TicTacToe RL game ! Choose your agent:
\t - 0 to train the epsilon-greedy algorithm
'''))
            if value == 0:
                os.system('clear')
                print("Training the epsilon greedy agent !")
                player_1 = agents.eps_greedy_agent(1,epsilon=0.3)
                player_2 = agents.eps_greedy_agent(-1,epsilon=0.3)
                environment = tictactoe(player_1, player_2)
                environment.play(1)
                break
            else:
                raise ValueError
        except ValueError:
            print("Invalid input :'( Try again")
            pass