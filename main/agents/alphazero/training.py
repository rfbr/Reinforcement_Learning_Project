import os
import pickle
import numpy as np
import torch
import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from main.agents.alphazero.net import Net, CustomLoss
from main.agents.alphazero.utils import load_results, load_state, board_data, save_data


def train_net(batch_size, epochs, iteration):
    data_path = "./main/agents/alphazero/data/datasets/iter_%d/" % iteration
    datasets = []
    for _, file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path, file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))
    datasets = np.array(datasets)
    net = Net()
    if torch.cuda.is_available():
        net.cuda()
    optimizer = torch.optim.Adam(net.parameters())
    start_epoch = load_state(net, iteration)

    train(net, datasets, optimizer, start_epoch, batch_size, epochs, iteration)


def train(net, dataset, optimizer, start_epoch, batch_size, epochs, iteration):

    net.train()
    cuda_availability = torch.cuda.is_available()
    criterion = CustomLoss()

    train_set = board_data(dataset)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    losses_per_epoch = load_results(iteration + 1)

    update_size = len(train_loader) // 10
    print("Update step size: %d" % update_size)
    print("Training the model...")
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        losses_per_batch = []
        for i, data in enumerate(train_loader, 0):
            state, policy, value = data
            state, policy, value = state.float(), policy.float(), value.float()
            if cuda_availability:
                state, policy, value = state.cuda(), policy.cuda(), value.cuda(
                )
            policy_pred, value_pred = net(state)
            loss = criterion(value_pred[:, 0], value, policy_pred, policy)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if i % update_size == (update_size - 1):
                losses_per_batch.append(total_loss / update_size)
                print(
                    f'[Iteration {iteration}] [Epoch: {epoch + 1}, {(i + 1) * batch_size}/ {len(train_set)} points] total loss per batch: {losses_per_batch[-1]}'
                )
                # print(f'Policy data: {policy[0]}')
                # print(f'Policy pred: {policy_pred[0]}')
                # print(f'Value (actual, predicted): {value[0].item()}, {value_pred[0, 0].item()}')
                print(" ")
                total_loss = 0.0
        if len(losses_per_batch) >= 1:
            losses_per_epoch.append(
                sum(losses_per_batch) / len(losses_per_batch))
        if (epoch % 2) == 0:
            save_data(
                losses_per_epoch,
                "./main/agents/alphazero/data/model_data/losses_per_epoch_iter%d.pkl"
                % (iteration + 1))
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    # 'optimizer': optimizer.state_dict()
                },
                os.path.join("./main/agents/alphazero/data/model_data/",
                             "%s_iter%d.pt" % ('AlphaZero', (iteration + 1))))

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(
        [e for e in range(start_epoch, (len(losses_per_epoch) + start_epoch))],
        losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")

    plt.savefig(
        os.path.join(
            "./main/agents/alphazero/data/model_data/",
            "Loss_vs_Epoch_iter%d_%s.png" %
            ((iteration + 1), datetime.datetime.today().strftime("%Y-%m-%d"))))
    plt.close()
