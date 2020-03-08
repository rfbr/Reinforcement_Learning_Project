# Reinforcement Learning

>**Emma DEMARECAUX - Charles DESALEUX - Romain FABRE**
************************
>*Comparing the performances of the _AlphaZero_ algorithm against other agents playing the tic-tac-toe game*

We implemented a bot that plays tic-tac-toe. It is a game for two players, `X` and `O`, who take turns marking the spaces in a 3 x 3 grid. 
The player who succeeds in placing three of their marks in a horizontal, vertical, or diagonal row is the winner.

As in the article [Silver, Hubert, et al. "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." 
*CoRR,abs/1712.01815.* (2017)](http://arxiv.org/abs/1712.01815), our idea is to compare the performances of several algorithms playing tic-tac-toe and to show that **AlphaZero** outperforms all of them.


The file `tic_tac_toe.py` contains the environment of the game and allows an algorithm to train using self-play or to play against another agent.
The file `__main__.py` is the main programme and is used to launch a competitions of several games between to agent. 
Once a competition is launched, the user is asked to chose the player 1 among:
- the random algorithm;
- the _epsilon-greedy_ algorithm;
- the _SARSA_ algorithm;
- the _Q-Learning_ algorithm;
- the _Expected SARSA_ algorithm;
- the _AlphaZero_ algorithm.

Then, the user selects a second player and decides the hyperparameters of both chosen agents (for instance the value of epsilon for the SARSA).
Finally, the user is asked the number of games for the competition.
Before playing the games, _SARSA_, _Expected SARSA_, _Q-Learning_ and _epsilon-greedy_ agents have to perform a training phase composed of games of self-play.
As the _epsilon-greedy_ takes a lot of time to train, its optimal policy at the end of the training will be stored in the `policies` folder to avoid future training of this agent with the same epsilon parameter.
The _AlphaZero_ agent is pre-trained, as the training take several hours to ensure good performances.

The results of the competition between the two trained agents is of the form (for 100 games):

| Agent                       | Win | Draw | Loss |
|-----------------------------|-----|------|------|
| Player 1: _AlphaZero_       | 0   | 100  | 0    |
| Player 2: Any other agent   | 0   | 100  | 0    |

in the case of good algorithms.

The folder `agents` contains all the tic-tac-toe algorithms that we implemented. _SARSA_, _Expected SARSA_, _Q-Learning_ come form the same base agent `value_agent.py` 
as their algorithms rely on the value function.

To launch the program, open a terminal and run from the root directory
```
python -m main
``` 
