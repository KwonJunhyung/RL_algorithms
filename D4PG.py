
"""
Author: David Valencia
Date: 13/ 05 /2022

Completed :
Describer: Still working here....
"""

import gym
import numpy as np

import torch
import torch.optim as optim

from Networks import Critic, Actor
from Memory   import MemoryClass

import matplotlib.pyplot as plt


class D4PGAgent:

    def __init__(self, env, actor_learning_rate=1e-4, critic_learning_rate=1e-4, gamma=0.99,
                 max_memory_size=50000, tau=1e-3, n_steps=1, batch_size=64):

        # -------- Hyper-parameters --------------- #
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size

        self.LEARN_EVERY_N_STEP = 50
        self.t_step = 0  # counter for activating learning step

        self.gamma   = gamma    # discount factor
        self.tau     = tau
        self.n_steps = n_steps
        self.hidden_size_critic = [512, 512, 256]
        self.hidden_size_actor  = [256, 256, 256]

        # -------- Parameters --------------- #
        self.num_states = env.observation_space.shape[0]  # 3
        self.num_actions = env.action_space.shape[0]      # 1

        # these parameters are used for the probability distribution
        self.n_atoms = 51
        self.v_min   = -10
        self.v_max   = 10
        self.delta   = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.v_lin   = torch.linspace(self.v_min, self.v_max, self.n_atoms).view(-1, 1)

        # ---------- Initialization and build the networks ----------- #
        # Main networks
        self.actor  = Actor(self.num_states, self.hidden_size_actor, self.num_actions)  # Actor net
        self.critic = Critic(self.num_states + self.num_actions, self.hidden_size_critic, self.n_atoms)  # Critic net

        # Target networks
        self.actor_target  = Actor(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, self.hidden_size_critic, self.n_atoms)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # optimizers
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # ------------- Initialization memory --------------------- #
        self.memory = MemoryClass(max_memory_size)


    def distr_projection(self, next_distribution, rewards, dones):
        next_distr = next_distribution.data.cpu().numpy()
        rewards    = rewards.data.cpu().numpy()
        dones_mask = dones.cpu().numpy().astype(bool)
        batch_size = len(rewards)
        proj_distr = np.zeros((batch_size, self.n_atoms), dtype=np.float32)
        gamma      = self.gamma ** self.n_steps

        for atom in range(self.n_atoms):
            tz_j = np.minimum(self.v_max, np.maximum(self.v_min, rewards + (self.v_min + atom * self.delta) * gamma))
            b_j = (tz_j - self.v_min) / self.delta
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l
            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

        if dones_mask.any():
            proj_distr[dones_mask] = 0.0
            tz_j = np.minimum(self.v_max, np.maximum(self.v_min, rewards[dones_mask]))
            b_j = (tz_j - self.v_min) / self.delta
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            eq_dones = dones_mask.copy()
            eq_dones[dones_mask] = eq_mask
            if eq_dones.any():
                proj_distr[eq_dones, l[eq_mask]] = 1.0
            ne_mask = u != l
            ne_dones = dones_mask.copy()
            ne_dones[dones_mask] = ne_mask
            if ne_dones.any():
                proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
                proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

        return torch.FloatTensor(proj_distr)


    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # numpy to a tensor with shape [1,3]
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.forward(state_tensor)
            action = action.detach()
            action = action.numpy()  # tensor to numpy
            noise  = np.random.normal(size=action.shape)
            action = np.clip(action + noise, -1, 1)  # todo maybe I could change this to -2, 2?
        self.actor.train()
        return action[0]

    def step_training(self, state, action, reward, next_state, done):
        # Save experience in memory
        self.memory.replay_buffer_add(state, action, reward, next_state, done)

        # learn step i.e. how often to learn and update networks
        self.t_step = self.t_step + 1
        if self.t_step % self.LEARN_EVERY_N_STEP == 0:
            self.learn_step()

    def learn_step(self):
        # check, if enough samples are available in memory
        if self.memory.__len__() <= self.batch_size:
            return
        else:
            states, actions, rewards, next_states, dones = self.memory.sample_experience(self.batch_size)

        states  = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        next_states = np.array(next_states)

        states  = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones   = torch.ByteTensor(dones)
        next_states = torch.FloatTensor(next_states)

        # calculate the next Z distribution Z(s',a') --> Q_next_value
        next_actions = self.actor_target.forward(next_states)  # Note this is from actor-target
        next_Z_val   = self.critic_target.forward(next_states, next_actions.detach())

        proj_distr_v = self.distr_projection(next_Z_val, rewards, dones)
        Y = proj_distr_v  # target_z_projected

        # calculate the distribution prediction Z(s,a) --> Q_values
        Z_val = self.critic.forward(states, actions)  # this is a categorical distribution, the Z predict

        # ----------------------------------- Calculate the loss ----- #
        # ------- calculate the critic loss
        BCE_loss = torch.nn.BCELoss(reduction='none')
        td_error = BCE_loss(Z_val, Y)
        td_error = td_error.mean(axis=1)
        critic_loss = td_error.mean()

        # ------- calculate the actor loss
        z_atoms = np.linspace(self.v_min, self.v_max, self.n_atoms)
        z_atoms = torch.from_numpy(z_atoms).float()
        actor_loss = self.critic.forward(states, self.actor.forward(states))
        actor_loss = actor_loss * z_atoms
        actor_loss = torch.sum(actor_loss, dim=1)
        actor_loss = -actor_loss.mean()


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


def main():
    EPISODES   = 100000  # ---> T, total number of episodes
    batch_size = 64

    env = gym.make('Pendulum-v1')
    agent = D4PGAgent(env, batch_size=batch_size)

    rewards     = []
    avg_rewards = []

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            #env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.step_training(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break
        print(f"******* -----Episode {episode + 1} Ended-----********* ")
        print("Episode total reward:", episode_reward)
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


if __name__ == '__main__':
    main()
