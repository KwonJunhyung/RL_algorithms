import torch
import torch.nn as nn


latent_dim = 512


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Actor(nn.Module):
    def __init__(self, action_dim, img_stack):
        super(Actor, self).__init__()
        self.encoder = torch.nn.ModuleList([  ## input size:[96, 96]
            torch.nn.Conv2d(img_stack, 16, 5, 2, padding=2),  ## output size: [16, 48, 48]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 5, 2, padding=2),  ## output size: [32, 24, 24]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 5, 2, padding=2),  ## output size: [64, 12, 12]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 5, 4, padding=2),  ## output size: [128, 3, 3]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),  ## output size: [256, 2, 2]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 512, 5, 2, padding=2),  ## output size: [512, 1, 1]
            Flatten(),  ## output: 512
        ])

        self.linear = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, action_dim),
            torch.nn.Tanh(),
        ])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        for layer in self.linear:
            x = layer(x)

        return x


class Critic(nn.Module):
    def __init__(self, action_dim, img_stack):
        super(Critic, self).__init__()
        self.encoder_1 = torch.nn.ModuleList([  ## input size:[96, 96]
            torch.nn.Conv2d(img_stack, 16, 5, 2, padding=2),  ## output size: [16, 48, 48]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 5, 2, padding=2),  ## output size: [32, 24, 24]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 5, 2, padding=2),  ## output size: [64, 12, 12]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 5, 4, padding=2),  ## output size: [128, 3, 3]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),  ## output size: [256, 2, 2]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 512, 5, 2, padding=2),  ## output size: [512, 1, 1]
            Flatten(),  ## output: 512
        ])

        self.encoder_2 = torch.nn.ModuleList([  ## input size:[96, 96]
            torch.nn.Conv2d(img_stack, 16, 5, 2, padding=2),  ## output size: [16, 48, 48]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 5, 2, padding=2),  ## output size: [32, 24, 24]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 5, 2, padding=2),  ## output size: [64, 12, 12]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 5, 4, padding=2),  ## output size: [128, 3, 3]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),  ## output size: [256, 2, 2]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 512, 5, 2, padding=2),  ## output size: [512, 1, 1]
            Flatten(),  ## output: 512
        ])

        self.linear_1 = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim + action_dim, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 1),
        ])

        self.linear_2 = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim + action_dim, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 1),
        ])

    def forward(self, x, u):
        x1 = x
        for layer in self.encoder_1:
            x1 = layer(x1)
        counter = 0
        for layer in self.linear_1:
            counter += 1
            if counter == 1:
                x1 = torch.cat([x1, u], 1)
                x1 = layer(x1)
            else:
                x1 = layer(x1)

        x2 = x
        for layer in self.encoder_2:
            x2 = layer(x2)
        counter = 0
        for layer in self.linear_2:
            counter += 1
            if counter == 1:
                x2 = torch.cat([x2, u], 1)
                x2 = layer(x2)
            else:
                x2 = layer(x2)

        return x1, x2

    def Q1(self, x, u):
        for layer in self.encoder_1:
            x = layer(x)

        counter = 0
        for layer in self.linear_1:
            counter += 1
            if counter == 1:
                x = torch.cat([x, u], 1)
                x = layer(x)
            else:
                x = layer(x)

        return x
