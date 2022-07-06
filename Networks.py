import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, num_atoms):
        super(Critic, self).__init__()

        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=num_atoms)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)   # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = self.h_linear_4(x)                  # No activation function here
        x = F.softmax(x, dim=1)                 # Softmax because critic should output probabilities
        return x


class CriticDDPG(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(CriticDDPG, self).__init__()

        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=num_actions)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)   # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = self.h_linear_4(x)                  # No activation function here
        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()

        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=output_size)

        self.max_action = 2.0  # todo fix get this value automatically somehow

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = torch.tanh(self.h_linear_4(x)) * self.max_action
        return x



class CriticTD3(nn.Module):

    def __init__(self, input_size, hidden_size, num_actions):
        super(CriticTD3, self).__init__()

        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=num_actions)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)   # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = self.h_linear_4(x)                  # No activation function here
        return x


class ActorTD3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorTD3, self).__init__()

        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.bn1 = nn.BatchNorm1d(hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=output_size)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.bn1(self.h_linear_3(x)))
        x = torch.tanh(self.h_linear_4(x))
        return x
