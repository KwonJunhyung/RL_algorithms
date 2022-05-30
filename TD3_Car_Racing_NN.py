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


class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()

        self.encoder_critic_1 = nn.Sequential(
            nn.Conv2d(4, 16, 5, 2, padding=2),  ## output size: [16, 48, 48]
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
        )

        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
        )

        self.encoder_critic_2 = nn.Sequential(
            nn.Conv2d(4, 16, 5, 2, padding=2),  ## output size: [16, 48, 48]
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
        )

        self.fc2 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
        )

    def forward(self, x, a):
        x1 = x
        x1 = self.encoder_critic_1(x1)
        x1 = torch.cat([x1, a], dim=1)
        x1 = self.fc1(x1)

        x2 = x
        x2 = self.encoder_critic_2(x2)
        x2 = torch.cat([x2, a], dim=1)
        x2 = self.fc2(x2)

        return x1, x2


    def Q1(self, x, a):
        x = self.encoder_critic_1(x)
        x = torch.cat([x, a], dim=1)
        x = self.fc1(x)
        return x
