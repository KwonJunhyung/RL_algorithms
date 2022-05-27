import gym
from TD3_NN import Actor, Critic
import torch
import numpy as np
from gym.envs.box2d.car_dynamics import Car
from PIL import Image, ImageDraw, ImageOps
import itertools


class TD3(object):
    def __init__(self, action_dim, img_stack):

        # ---------- Initialization and build the networks ----------- #
        self.action_dim = action_dim
        self.actor = Actor(action_dim, img_stack)
        self.actor_target = Actor(action_dim, img_stack)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.actor_loss = []

        self.critic = Critic(action_dim, img_stack)
        self.critic_target = Critic(action_dim, img_stack)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.critic_loss = []

    def select_action(self, state):
        state = np.array(state)
        state = torch.FloatTensor(state).permute(1, 0, 2, 3)
        state = state.float()
        action = self.actor.forward(state).cpu().data.numpy().flatten()
        return action



def stack_frames(frame):
    pass


def rgb2gray(rgb, norm=True):
    # rgb image -> gray [0, 1]
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray


def reset_states(env, img_stack):
    initial_frame = env.reset()
    img_gray = rgb2gray(initial_frame)
    stack_img = [np.expand_dims(img_gray, axis=0)] * img_stack  # four frames
    #print(np.shape(stack_img))  # (4, 1, 96, 96)
    return stack_img


def take_step(env, action):
    img_rgb, reward, die, _ = env.step(action)



def main():
    num_episodes     = 1
    episode_horizont = 1
    batch_size = 32

    env = gym.make('CarRacing-v1')
    action_dim = env.action_space.shape[0]  # 3
    img_stack  = 4  # number of image stacks together
    agent = TD3(action_dim, img_stack)

    for episode in range(num_episodes):
        state = reset_states(env, img_stack)

        for t in range(episode_horizont):
            action = agent.select_action(state)
            take_step(env, action)











if __name__ == '__main__':
    main()
