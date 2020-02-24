import os
from main.agents import eps_greedy
from main.tic_tac_toe import TicTacToe

if __name__ == '__main__':
    players = {}
    os.system('clear')
    while True:
        try:
            print('Welcome to the tic-tac-toe RL game!')
            player_1_value = int(input(
                ''' Choose player 1 agent:
        - 0 to play with the epsilon-greedy algorithm
                \n'''
            ))
            player_2_value = int(input(
                '''Choose player 2 agent:
        - 0 to play with the epsilon-greedy algorithm
                \n'''
            ))
            if player_1_value == 0:
                os.system('clear')
                while True:
                    try:
                        eps_1 = float(input('Player 1 epsilon value?\n'))
                        if eps_1 < 0 or eps_1 > 1:
                            raise ValueError
                        else:
                            players[1] = eps_greedy.EpsGreedyAgent(
                                name=1, epsilon=eps_1)
                            break
                    except ValueError:
                        print('Epsilon must be in (0,1)')

            if player_2_value == 0:
                os.system('clear')
                while True:
                    try:
                        eps_2 = float(input('Player 2 epsilon value?\n'))
                        if eps_2 < 0 or eps_2 > 1:
                            raise ValueError
                        else:
                            players[2] = eps_greedy.EpsGreedyAgent(
                                name=-1, epsilon=eps_2)
                            break
                    except ValueError:
                        print('Epsilon must be in (0,1)\n')

            os.system('clear')
            while True:
                try:
                    nb_games = int(
                        input('How many games you want them to play?\n'))
                    if nb_games <= 0:
                        raise ValueError
                    else:
                        break
                except ValueError:
                    print('Oops wrong input !')

        except ValueError:
            print("Invalid input :'( Try again")

        environment = TicTacToe(players[1], players[2], nb_games)
        environment.play()

        break
