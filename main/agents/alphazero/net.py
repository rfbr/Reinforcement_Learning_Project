import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import main.agents.alphazero.mcts as mcsts


class InputBlock(nn.Module):
    def __init__(self, input_channels, n0, kernel_size=3):
        super(InputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=input_channels,
                                out_channels=n0,
                                kernel_size=kernel_size,
                                padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(num_features=n0)

    def forward(self, x):
        x = x.view(-1, 2, 3, 3)
        output = F.relu(self.batch_norm_1(self.conv_1(x)))
        return output


class ResidualBlock(nn.Module):
    def __init__(self, n0, n1, n2, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=n0,
                                out_channels=n1,
                                kernel_size=kernel_size,
                                padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(num_features=n1)
        self.conv_2 = nn.Conv2d(in_channels=n1,
                                out_channels=n2,
                                kernel_size=kernel_size,
                                padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=n2)

    def forward(self, x):
        residual_connection = x
        temp = F.relu(self.batch_norm_1(self.conv_1(x)))
        temp = self.batch_norm_2(self.conv_2(temp))
        temp += residual_connection
        output = F.relu(temp)
        return output


class PolicyOutputBlock(nn.Module):
    def __init__(self, n0, n1, kernel_size=3):
        super(PolicyOutputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=n0,
                                out_channels=n1,
                                kernel_size=kernel_size,
                                padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(n1)
        self.fc = nn.Linear(3 * 3 * n1, 9)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        policy = F.relu(self.batch_norm_1(self.conv_1(x)))
        policy = policy.view(-1, 3 * 3 * 32)
        policy = self.fc(policy)
        policy = self.logsoftmax(policy).exp()
        return policy


class ValueOutputBlock(nn.Module):
    def __init__(self, n0, n1, kernel_size=1, padding=1):
        super(ValueOutputBlock, self).__init__()
        self.conv = nn.Conv2d(n0, n1, kernel_size=1)
        self.bn = nn.BatchNorm2d(n1)
        self.fc1 = nn.Linear(3 * 3 * n1, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        value = F.relu(self.bn(self.conv(x)))
        value = value.view(-1, 3 * 3 * 3)
        value = F.relu(self.fc1(value))
        value = torch.tanh(self.fc2(value))
        return value


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value)**2
        policy_error = torch.sum(
            (-policy * (1e-8 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error


class Net(nn.Module):
    def __init__(self, name=None):
        super(Net, self).__init__()
        self.input_conv = InputBlock(input_channels=2, n0=128)
        for block in range(4):
            setattr(self, "res_%i" % block,
                    ResidualBlock(n0=128, n1=128, n2=128))
        self.policy_output = PolicyOutputBlock(n0=128, n1=32)
        self.value_output = ValueOutputBlock(n0=128, n1=3)
        self.name = name
        self.display_name = 'AlphaZero'
        self.states = []

    def forward(self, x):
        x = self.input_conv(x)
        for block in range(4):
            x = getattr(self, "res_%i" % block)(x)
        p = self.policy_output(x)
        v = self.value_output(x)
        return p, v

    def action(self, board):
        root = mcsts.mcts_simulation(board, 100, self, self.name)
        policy = mcsts.compute_policy(root)
        move = np.random.choice(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]), 1, p=policy)
        action = (move // 3, move % 3)
        return action

    def clear_states(self):
        self.states = []
