import time

import gym
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable

from Networks import CriticDDPG, Actor
from Memory import MemoryClass
from Noise import OUNoise

import matplotlib.pyplot as plt


class DDPGagent:

    def __init__(self, env, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99,
                 max_memory_size=50000, tau=0.005, batch_size=64):

        # -------- Hyper-parameters --------------- #
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size

        self.update_interaction = 1  # number of updates of NN per Episode

        self.gamma = gamma  # discount factor
        self.tau = tau
        self.hidden_size_critic = [512, 512, 256]
        self.hidden_size_actor  = [256, 256, 256]

        # -------- Parameters --------------- #
        self.num_states = env.observation_space.shape[0]  # 3
        self.num_actions = env.action_space.shape[0]  # 1

        # ---------- Initialization and build the networks ----------- #
        # Main networks
        self.actor = Actor(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic = CriticDDPG(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

        # Target networks
        self.actor_target  = Actor(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic_target = CriticDDPG(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # ------------- Initialization memory --------------------- #
        self.memory = MemoryClass(max_memory_size)

    def get_action(self, state):

        # todo try with this too
        '''
        state_chan = torch.from_numpy(state).float().unsqueeze(0)  # a tensor with shape [1,3]
        state_ready = Variable(state_chan)  # need Variable to perform backpropagation
        action = self.actor.forward(state_ready)
        action = action.detach()
        action = action.numpy()
        return action[0]
        '''


        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # numpy to a tensor with shape [1,3]
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
        # update the networks every N times
        for it in range(self.update_interaction):
            states, actions, rewards, next_states, dones = self.memory.sample_experience(self.batch_size)

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards).reshape(-1, 1)
            dones = np.array(dones).reshape(-1, 1)
            next_states = np.array(next_states)

            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)
            next_states = torch.FloatTensor(next_states)

            # ----------------------------------- Calculate the loss ----- #
            Q_vals = self.critic.forward(states, actions)

            # ------- calculate the critic loss
            next_actions = self.actor_target.forward(next_states)  # Note this is from actor-target

            # next_Q_vales = self.critic_target.forward(next_states, next_actions.detach())
            # Q_target = rewards + self.gamma * next_Q_vales

            next_Q_vales = self.critic_target.forward(next_states, next_actions)
            Q_target = rewards + (self.gamma * next_Q_vales).detach()

            loss = nn.MSELoss()
            critic_loss = loss(Q_vals, Q_target)

            # ------- calculate the actor loss
            actor_loss = - self.critic.forward(states, self.actor.forward(states)).mean()

            # ------------------------------------- Update main networks --------------- #
            # Actor step Update
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Critic step Update
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ------------------------------------- Update target networks --------------- #
            # update the target networks using tao "soft updates"
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_model(self):
        torch.save(self.actor.state_dict(), 'weights/ddpg_actor_1K.pth')
        torch.save(self.critic.state_dict(), 'weights/ddpg_critic_1K.pth')
        print("models has been saved...")

    def load_model(self):
        self.actor.load_state_dict(torch.load('weights/ddpg_actor_1K.pth'))
        self.critic.load_state_dict(torch.load('weights/ddpg_critic_1K.pth'))
        print("models has been loaded...")



def main():
    EPISODES        = 10000  # Total number of episodes
    render_interval = EPISODES * 0.95  # Activate render after 95% of total episodes
    batch_size      = 64

    env = gym.make('Pendulum-v1')
    agent = DDPGagent(env, batch_size=batch_size)
    noise = OUNoise(env.action_space)

    rewards = []
    avg_rewards = []

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        noise.reset()
        episode_reward = 0
        step = 0
        while not done:
            #if episode >= render_interval:env.render()
            action = agent.get_action(state)
            action = noise.get_action(action, step)
            new_state, reward, done, _ = env.step(action)
            agent.add_experience_memory(state, action, reward, new_state, done)
            state = new_state
            if done:
                agent.step_training()  # Update the NNs
                #break
            step += 1
            episode_reward += reward
        print(f"******* -----Episode {episode} Ended-----********* ")
        print("Episode total reward:", episode_reward)
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

    agent.save_model()

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig("DDPG.png")
    #plt.show()


def test_load():
    TEST_EPISODES = 10
    env = gym.make('Pendulum-v1')
    agent = DDPGagent(env)
    agent.load_model()
    for episode in range(1, TEST_EPISODES + 1):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            env.render()
            action = agent.get_action(state)
            #action = env.action_space.sample()
            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = new_state
            time.sleep(0.03)
            if done:
                break
        print("Episode total reward:", episode_reward)


if __name__ == '__main__':
    main()
    #test_load()
