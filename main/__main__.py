import os
from main.agents.sarsa import SARSA
from main.agents.eps_greedy import EpsGreedyAgent
from main.agents.q_learning import QLearning
from main.tic_tac_toe import TicTacToe

if __name__ == '__main__':
    players = {}
    os.system('clear')
    while True:
        try:
            print('Welcome to the tic-tac-toe RL game!')
            possible_agents = '''
        - 0 to play with the random algorithm
        - 1 to play with the epsilon-greedy algorithm
        - 2 to play with the SARSA algorithm
        - 3 to play with the Q Learning algorithm
        '''
            possible_choices = [0, 1, 2, 3]
            # -- Choices of players agents
            while True:
                try:
                    player_1_value = int(input("Choose player 1 agent:" + possible_agents + "\n"))
                    if player_1_value not in possible_choices:
                        raise ValueError
                    else:
                        break
                except ValueError:
                    print('Player 1 agent must be in ', possible_choices)
            while True:
                try:
                    player_2_value = int(input("Choose player 2 agent:" + possible_agents + "\n"))
                    if player_2_value not in possible_choices:
                        raise ValueError
                    else:
                        break
                except ValueError:
                    print('Player 2 agent must be in ', possible_choices)
            # -- Player initialisation
            p1_need_training = False
            p2_need_training = False
            # - Player 1
            # Random algorithm
            if player_1_value == 0:
                os.system('clear')
                players[1] = EpsGreedyAgent(name=1, epsilon=1)
            # Epsilon-greedy algorithm
            if player_1_value == 1:
                os.system('clear')
                while True:
                    try:
                        eps_1 = float(input('Player 1 epsilon value?\n'))
                        if eps_1 < 0 or eps_1 >= 1:
                            raise ValueError
                        else:
                            players[1] = EpsGreedyAgent(name=1, epsilon=eps_1)
                            p1_policy_name = 'p1_epsilon_' + str(eps_1)
                            try:
                                players[1].load_policy("main/policies/" + p1_policy_name)
                            except (OSError, IOError) as e:
                                p1_need_training = True
                                env1 = TicTacToe(players[1], EpsGreedyAgent(name=-1, epsilon=eps_1))
                            break
                    except ValueError:
                        print('Epsilon must be in [0,1[')
            # SARSA algorithm
            if player_1_value == 2:
                os.system('clear')
                while True:
                    try:
                        eps_1 = float(input('Player 1 epsilon value?\n'))
                        if eps_1 < 0 or eps_1 > 1:
                            raise ValueError
                        else:
                            break
                    except ValueError:
                        print('Epsilon must be in [0,1]')
                while True:
                    try:
                        alpha_1 = float(input('Player 1 alpha value?\n(default is 0.8)\n'))
                        if alpha_1 < 0 or alpha_1 > 1:
                            raise ValueError
                        else:
                            break
                    except ValueError:
                        print('Alpha must be in [0,1]')
                players[1] = SARSA(name=1, epsilon=eps_1, alpha=alpha_1)
                p1_need_training = True
                env1 = TicTacToe(players[1], SARSA(name=-1, epsilon=eps_1, alpha=alpha_1))
                p1_policy_name = ''
            # Q Learning algorithm
            if player_1_value == 3:
                os.system('clear')
                while True:
                    try:
                        eps_1 = float(input('Player 1 epsilon value?\n'))
                        if eps_1 < 0 or eps_1 > 1:
                            raise ValueError
                        else:
                            break
                    except ValueError:
                        print('Epsilon must be in [0,1]')
                while True:
                    try:
                        alpha_1 = float(input('Player 1 alpha value?\n(default is 0.8)\n'))
                        if alpha_1 < 0 or alpha_1 > 1:
                            raise ValueError
                        else:
                            break
                    except ValueError:
                        print('Alpha must be in [0,1]')
                players[1] = QLearning(name=1, epsilon=eps_1, alpha=alpha_1)
                p1_need_training = True
                env1 = TicTacToe(players[1], QLearning(name=-1, epsilon=eps_1, alpha=alpha_1))
                p1_policy_name = ''
            # - Player 2
            # Random algorithm
            if player_2_value == 0:
                os.system('clear')
                players[2] = EpsGreedyAgent(name=-1, epsilon=1)
            # Epsilon-greedy algorithm
            if player_2_value == 1:
                os.system('clear')
                while True:
                    try:
                        eps_2 = float(input('Player 2 epsilon value?\n'))
                        if eps_2 < 0 or eps_2 >= 1:
                            raise ValueError
                        else:
                            players[2] = EpsGreedyAgent(name=-1, epsilon=eps_2)
                            p2_policy_name = 'p2_epsilon_' + str(eps_2)
                            try:
                                players[2].load_policy("main/policies/" + p2_policy_name)
                            except (OSError, IOError) as e:
                                p2_need_training = True
                                env2 = TicTacToe(EpsGreedyAgent(name=1, epsilon=eps_2), players[2])
                            break
                    except ValueError:
                        print('Epsilon must be in [0,1[')
            # SARSA algorithm
            if player_2_value == 2:
                os.system('clear')
                while True:
                    try:
                        eps_2 = float(input('Player 2 epsilon value?\n'))
                        if eps_2 < 0 or eps_2 > 1:
                            raise ValueError
                        else:
                            break
                    except ValueError:
                        print('Epsilon must be in [0,1]')
                while True:
                    try:
                        alpha_2 = float(input('Player 2 alpha value?\n(default is 0.8)\n'))
                        if alpha_2 < 0 or alpha_2 > 1:
                            raise ValueError
                        else:
                            break
                    except ValueError:
                        print('Alpha must be in [0,1]')
                players[2] = SARSA(name=-1, epsilon=eps_2, alpha=alpha_2)
                p2_need_training = True
                env2 = TicTacToe(SARSA(name=1, epsilon=eps_2, alpha=alpha_2), players[2])
                p2_policy_name = ''
            # Q Learning algorithm
            if player_2_value == 3:
                os.system('clear')
                while True:
                    try:
                        eps_2 = float(input('Player 2 epsilon value?\n'))
                        if eps_2 < 0 or eps_2 > 1:
                            raise ValueError
                        else:
                            break
                    except ValueError:
                        print('Epsilon must be in [0,1]')
                while True:
                    try:
                        alpha_2 = float(input('Player 2 alpha value?\n(default is 0.8)\n'))
                        if alpha_2 < 0 or alpha_2 > 1:
                            raise ValueError
                        else:
                            break
                    except ValueError:
                        print('Alpha must be in [0,1]')
                players[2] = QLearning(name=-1, epsilon=eps_2, alpha=alpha_2)
                p2_need_training = True
                env2 = TicTacToe(QLearning(name=1, epsilon=eps_2, alpha=alpha_2), players[2])
                p2_policy_name = ''
            os.system('clear')
            while True:
                try:
                    nb_games = int(input('How many games you want them to play?\n'))
                    if nb_games <= 0:
                        raise ValueError
                    else:
                        break
                except ValueError:
                    print('Oops wrong input!')
        except ValueError:
            print("Invalid input :'( Try again")
        print("Training in progress...")
        if p1_need_training:
            env1.train(1000)
            env1.player_1.save_policy("main/policies/" + p1_policy_name)
            players[1].load_policy("main/policies/" + p1_policy_name)
        if p2_need_training:
            env2.train(1000)
            env2.player_2.save_policy("main/policies/" + p2_policy_name)
            players[2].load_policy("main/policies/" + p2_policy_name)
        print("Playing games...")
        environment = TicTacToe(players[1], players[2])
        environment.simulation(nb_games)
        break
