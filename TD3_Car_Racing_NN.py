import torch
import torch.nn as nn

latent_dim = 512


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Actor(nn.Module):

    def __init__(self, action_dim):

        super(Actor, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 5, 2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 5, 2, padding=2),  ## output size: [32, 24, 24]
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 5, 2, padding=2),  ## output size: [64, 12, 12]
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 5, 4, padding=2),  ## output size: [128, 3, 3]
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 5, 2, padding=2),  ## output size: [256, 2, 2]
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 5, 2, padding=2),  ## output size: [512, 1, 1]
            Flatten(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 30),
            nn.ReLU(),
            nn.Linear(30, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        return x
