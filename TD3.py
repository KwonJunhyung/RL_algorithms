import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from Memory import MemoryClass
from Networks import CriticTD3, ActorTD3


class TD3agent(object):

    def __init__(self, env, batch_size=32, max_memory_size=10_000, gamma=0.99,
                 critic_learning_rate=1e-3, actor_learning_rate=1e-4, tau=0.005):

        # -------- Hyper-parameters --------------- #
        self.tau   = tau
        self.gamma = gamma   # discount factor
        self.max_memory_size = max_memory_size
        self.batch_size      = batch_size

        self.update_interaction    = 1
        self.interal_target_update = 2

        self.update_counter = 0

        self.hidden_size_critic = [512, 512, 256]
        self.hidden_size_actor  = [256, 256, 256]

        # -------- Parameters --------------- #
        self.num_states  = env.observation_space.shape[0]  # 3
        self.num_actions = env.action_space.shape[0]      # 1

        # ------------- Initialization memory --------------------- #
        self.memory = MemoryClass(max_memory_size)

        # ---------- Initialization and build the networks ----------- #
        # Main networks
        self.actor  = ActorTD3(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic = CriticTD3(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

        # Target networks
        self.actor_target  = ActorTD3(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic_target = CriticTD3(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)



    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # numpy to a tensor with shape [1,3]
        action = self.actor.forward(state_tensor)
        action = action.cpu().data.numpy()
        return action[0]


    def add_experience_memory(self, state, action, reward, next_state, done):
        # Save experience in memory
        self.memory.replay_buffer_add(state, action, reward, next_state, done)


    def step_training(self):

        # check, if enough samples are available in memory
        if self.memory.__len__() <= self.batch_size:
            return

        self.update_counter += 1

        # update the networks every N times
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
            dones = torch.FloatTensor(dones)
            next_states = torch.FloatTensor(next_states)

            Q_vals_q1, Q_vals_q2 = self.critic.forward(states, actions)

            with torch.no_grad():
                next_actions = self.actor_target.forward(next_states)
                noise_action = (torch.randn_like(actions) * 0.05).clamp(-1, 1)
                next_actions = (next_actions + noise_action).clamp(-1, 1)

                # compute next targets values
                next_Q_vales_q1, next_Q_vales_q2 = self.critic_target.forward(next_states, next_actions)
                q_min = torch.minimum(next_Q_vales_q1, next_Q_vales_q2)

                #Q_target = rewards + (self.gamma * (1 - dones) * q_min)
                Q_target = rewards + (self.gamma * q_min)


            loss = nn.MSELoss()
            critic_loss = loss(Q_vals_q1, Q_target) + loss(Q_vals_q2, Q_target)

            # Critic step Update
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


            # TD3 updates the policy (and target networks) less frequently than the Q-function
            if self.update_counter % self.interal_target_update == 0:

                # ------- calculate the actor loss
                actor_loss = - self.critic.Q1(states, self.actor.forward(states)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ------------------------------------- Update target networks --------------- #

                # update the target networks using tao "soft updates"
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic.parameters(), self.critic.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))



def main():

    EPISODES = 50_000  # Total number of episodes
    render_interval = EPISODES * 0.99  # Activate render after 95% of total episodes
    batch_size = 32

    env = gym.make('Pendulum-v1')
    agent = TD3agent(env, batch_size=batch_size)

    rewards     = []
    avg_rewards = []

    for episode in range(1, EPISODES + 1):
        
        print(f"-------Episode:{episode} ---------")

        state = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:

            #if episode >= render_interval:env.render()

            action = agent.get_action(state)
            noise  = (np.random.normal(0, scale=0.1, size=1))
            action = action + noise
            action = np.clip(action, -1, 1)

            convert_action = (action - (-1)) * (2 - (-2)) / (1 - (-1)) + (-2)  # to put the output between +2 -2

            new_state, reward, done, _ = env.step(convert_action)
            agent.add_experience_memory(state, action, reward, new_state, done)

            state = new_state

            if done:
                agent.step_training()  # Update the NNs
                break

            episode_reward += reward

        print(f"******* -----Episode {episode} Ended-----********* ")
        print("Episode total reward:", episode_reward)
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-100:]))


    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # plt.savefig("TD3.png")
    plt.show()


if __name__ == '__main__':
    main()
