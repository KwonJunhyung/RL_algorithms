
"""
    A random track is generated for every episode.
    The reward is -0.1 every frame and +1000/N for every track tile visited,
    where N is the total number of tiles in track
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageOps

from TD3_Car_Racing_Memory import TD3_Memory
from TD3_Car_Racing_NN import Actor, Critic
import matplotlib.pyplot as plt

import time
counter = 0


class TD3_Racing(object):

    def __init__(self, env, batch_size):

        self.critic_learning_rate = 1e-3
        self.actor_learning_rate  = 1e-4
        self.tau = 0.005

        self.gamma = 0.99
        self.max_memory_size = 10_000
        self.batch_size = batch_size
        self.action_dim = env.action_space.shape[0]  # 3

        self.update_interaction = 6
        self.policy_freq_update = 2
        self.update_counter     = 0


        # ------------- Initialization memory --------------------- #
        self.memory = TD3_Memory(self.max_memory_size)

        # ---------- Initialization and build the networks ----------- #
        # Main Networks
        self.actor     = Actor(self.action_dim)
        self.critic_q1 = Critic(self.action_dim)
        self.critic_q2 = Critic(self.action_dim)

        # Target networks
        self.actor_target  = Actor(self.action_dim)
        self.critic_target_q1 = Critic(self.action_dim)
        self.critic_target_q2 = Critic(self.action_dim)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_q1.parameters(), self.critic_q1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_q2.parameters(), self.critic_q2.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer    = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer_1 = optim.Adam(self.critic_q1.parameters(), lr=self.critic_learning_rate)
        self.critic_optimizer_2 = optim.Adam(self.critic_q2.parameters(), lr=self.critic_learning_rate)

    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.forward(state_tensor)
            action = action.detach()
            action = action.numpy()  # tensor to numpy
        self.actor.train()
        return action[0]


    def add_experience_memory(self, state, action, reward, next_state, done):
        # Save experience in memory
        self.memory.replay_buffer_add(state, action, reward, next_state, done)

    def step_training(self):
        # check, if enough samples are available in memory
        if self.memory.__len__() <= self.batch_size:
            return

        self.update_counter += 1

        for it in range(self.update_interaction):

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

            #print(states.size())       # --> [Batch_size, 4, 64, 64]
            #print(next_states.size())  # --> [Batch_size, 4, 64, 64]
            #print(actions.size())      # --> [Batch_size, 3]
            #print(rewards.size())      # --> [Batch_size, 1]
            #print(dones.size())        # --> [Batch_size, 1]

            # ------- compute the target action
            next_actions = self.actor_target.forward(next_states)

            # add noise also here, paper mentioned this
            next_actions = next_actions.detach().numpy()  # tensor to numpy
            next_actions = next_actions + (np.random.normal(0, scale=0.2, size=self.action_dim))
            next_actions = np.clip(next_actions, -1, 1)
            next_actions = torch.FloatTensor(next_actions)

            # compute next targets values
            next_Q_vales_q1 = self.critic_target_q1.forward(next_states, next_actions)
            next_Q_vales_q2 = self.critic_target_q2.forward(next_states, next_actions)

            q_min = torch.minimum(next_Q_vales_q1, next_Q_vales_q2)

            Q_target = rewards + (self.gamma * (1 - dones) * q_min).detach()

            loss = nn.MSELoss()

            Q_vals_q1 = self.critic_q1.forward(states, actions)
            Q_vals_q2 = self.critic_q2.forward(states, actions)

            critic_loss_1 = loss(Q_vals_q1, Q_target)
            critic_loss_2 = loss(Q_vals_q2, Q_target)

            # Critic step Update
            self.critic_optimizer_1.zero_grad()
            critic_loss_1.backward()
            self.critic_optimizer_1.step()

            self.critic_optimizer_2.zero_grad()
            critic_loss_2.backward()
            self.critic_optimizer_2.step()

            # TD3 updates the policy (and target networks) less frequently than the Q-function
            if self.update_counter % self.policy_freq_update == 0:
                # ------- calculate the actor loss
                actor_loss = - self.critic_q1.forward(states, self.actor.forward(states)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ------------------------------------- Update target networks --------------- #
                # update the target networks using tao "soft updates"
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_target_q1.parameters(), self.critic_q1.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_target_q2.parameters(), self.critic_q2.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_model(self):
        torch.save(self.actor.state_dict(), 'weights/td3_car_racing_actor.pth')
        torch.save(self.critic_q1.state_dict(), 'weights/td3_car_racing_critic_1.pth')
        torch.save(self.critic_q2.state_dict(), 'weights/td3_critic_cube_critic_2.pth')
        print("models has been saved...")

    def load_model(self):
        self.actor.load_state_dict(torch.load('weights/td3_car_racing_actor.pth'))
        self.critic_q1.load_state_dict(torch.load('weights/td3_car_racing_critic_1.pth'))
        self.critic_q2.load_state_dict(torch.load('weights/td3_critic_cube_critic_2.pth'))
        print("models has been loaded...")



def crop_normalize_observation(observation):
    image = Image.fromarray(observation[:83, :, :], mode='RGB')  # removing bottom of the image
    image = image.resize((64, 64), Image.BILINEAR)  # resizing to (64X64X3)
    gray_image = ImageOps.grayscale(image)  # 64x64
    img = np.asarray(gray_image)
    img = img.astype('float32')  # convert from integers to floats 32
    img = img / 255.0  # normalize the image [0~1]
    return img


def rgb2gray(rgb, norm=True):
    # rgb image -> gray [0, 1]
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray


def main():
    num_episodes     = 1000
    render_interval  = num_episodes * 0.99
    episode_horizont = 1000  # after this number env auto-reset
    batch_size       = 64

    env   = gym.make('CarRacing-v1')
    agent = TD3_Racing(env, batch_size)

    rewards     = []
    avg_rewards = []

    for episode in range(1, num_episodes+1):

        initial_frame = env.reset()
        initial_frame = crop_normalize_observation(initial_frame)

        state_frame_queue = deque([initial_frame] * 4, maxlen=4)
        episode_reward = 0
        done = False

        for step in range(episode_horizont):
            #if episode >= render_interval:env.render()

            current_state = np.array(state_frame_queue)  # (4,64,64)

            action = agent.get_action(current_state)
            action = action + (np.random.normal(0, scale=0.1, size=env.action_space.shape[0]))
            action = np.clip(action, -1, 1)
            next_frame, reward, done, _ = env.step(action)
            next_frame = crop_normalize_observation(next_frame)
            state_frame_queue.append(next_frame)
            next_state = np.array(state_frame_queue)

            if step > 50:
                agent.add_experience_memory(current_state, action, reward, next_state, done)

            episode_reward += reward

            if done:
                print("updating Networks")
                agent.step_training()
                break

        print(f"Episode {episode} Ended, Episode total reward: {episode_reward}")
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


if __name__ == '__main__':
    main()
