from main.agents.alphazero.mcts import run_mcts
from main.agents.alphazero.training import train_net
from main.agents.alphazero.evaluation import evaluate_nets

if __name__ == '__main__':
    iteration = 0
    total_iterations = 5
    batch_size = 64
    epochs = 100
    game_per_mcts = 100
    for i in range(iteration, total_iterations):
        run_mcts(game_per_mcts, 0, i)
        train_net(batch_size, epochs, i)
        if i >= 1:
            winner = evaluate_nets(i, i + 1)
            counts = 0
            while winner != (i + 1):
                run_mcts(game_per_mcts, (counts + 1) * game_per_mcts, i)
                counts += 1
                train_net(batch_size=batch_size, epochs=epochs, iteration=i)
                winner = evaluate_nets(i, i + 1)
