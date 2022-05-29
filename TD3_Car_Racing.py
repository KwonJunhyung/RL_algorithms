import gym
from TD3_NN import Actor, Critic
import torch
import numpy as np
from gym.envs.box2d.car_dynamics import Car
from PIL import Image, ImageDraw, ImageOps
import itertools

import time
from TD3_Car_Racing_Memory import TD3_Memory
from TD3_Car_Racing_NN import Actor



'''
class TD3(object):
    def __init__(self, action_dim, env):

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

'''


class TD3_Racing(object):

    def __init__(self, env, batch_size):

        self.max_memory_size = 10_000
        self.batch_size = batch_size
        self.action_dim = env.action_space.shape[0]

        # ------------- Initialization memory --------------------- #
        self.memory = TD3_Memory(self.max_memory_size)


        # ---------- Initialization and build the networks ----------- #
        # Main Networks
        self.actor = Actor(self.action_dim)

        # Target networks
        self.actor_target = Actor(self.action_dim)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)



    def add_experience_memory(self, state, action, reward, next_state, done):
        # Save experience in memory
        self.memory.replay_buffer_add(state, action, reward, next_state, done)

    def step_training(self):

        # check, if enough samples are available in memory
        if self.memory.__len__() <= self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_experience(self.batch_size)

        states  = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        dones   = np.array(dones).reshape(-1, 1)
        next_states = np.array(next_states)

        states  = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones   = torch.FloatTensor(dones)
        next_states = torch.FloatTensor(next_states)

        # ------- compute the target action
        print(states.shape)
        print(next_states.shape)

        next_actions = self.actor_target.forward(states)
        #print(next_actions)


def crop_normalize_observation(observation):
    crop_frame = Image.fromarray(observation[:83, :, :], mode='RGB')  # removing bottom of the image
    img_resize = crop_frame.resize((64, 64), Image.BILINEAR)  # resizing to (64X64X3)
    gray_image = ImageOps.grayscale(img_resize)
    img = np.asarray(gray_image)
    img = img.astype('float32')  # convert from integers to floats 32
    img = img / 255.0  # normalize the image [0~1]
    return img


def main():

    num_episodes = 10
    episode_horizont = 500
    batch_size = 5

    env = gym.make('CarRacing-v0')
    agent = TD3_Racing(env, batch_size)

    for episode in range(num_episodes):

        state = env.reset()
        state = crop_normalize_observation(state)
        done  = False
        score = 0

        for step in range(episode_horizont):
            # env.render()
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            env.viewer.window.dispatch_events()  # Avoiding possibles corrupt environment observations
            next_state = crop_normalize_observation(next_state)

            if step > 50:  # with > 50; I discard the first 50 states since they are useless, no info there
                agent.add_experience_memory(state, action, reward, next_state, done)

            state = next_state

            score += reward

            agent.step_training()











if __name__ == '__main__':
    main()
