# Based on When to Trust Your Model: Model-Based Policy Optimization

import gym
import numpy as np
import torch
import random
from collections import deque
import torch.nn as nn
import torch.optim as optim


class Memory:

    def __init__(self, replay_max_size):
        self.replay_max_size = replay_max_size
        self.replay_buffer_env   = deque(maxlen=replay_max_size)
        self.replay_buffer_model = deque(maxlen=replay_max_size)

    def replay_buffer_environment_add(self, state, action, reward, next_state, done):
        experience_from_env = (state, action, reward, next_state, done)
        self.replay_buffer_env.append(experience_from_env)

    def replay_buffer_model_add(self, state, action, reward, next_state, done):
        experience_from_model = (state, action, reward, next_state, done)
        self.replay_buffer_model.append(experience_from_model)

    def sample_experience_from_env(self, batch_size):

        state_batch      = []
        action_batch     = []
        reward_batch     = []
        done_batch       = []
        next_state_batch = []

        batch_of_experiences = random.sample(self.replay_buffer_env, batch_size)

        for experience in batch_of_experiences:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def len_env_buffer(self):
        return len(self.replay_buffer_env)

    def len_model_buffer(self):
        return len(self.replay_buffer_model)


class ModelNet_transitions(nn.Module):
    # This neural network uses action and current state as inputs.
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNet_transitions, self).__init__()
        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=output_size)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = self.h_linear_3(x)
        return x


class ModelNet_reward(nn.Module):
    # This neural network uses action and next state as inputs.
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNet_reward, self).__init__()
        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=output_size)

    def forward(self, next_state, action):
        x = torch.cat([next_state, action], dim=1)  # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = self.h_linear_3(x)
        return x


class ModelNet_N_done(nn.Module):
    #  This neural network uses only next state as inputs.
    def __init__(self, input_size):
        super(ModelNet_N_done, self).__init__()

        hidden_size = [200, 200]
        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=1)

    def forward(self, next_state):
        x = next_state
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = torch.sigmoid(self.h_linear_3(x))
        return x


class ModelAgent:

    def __init__(self, env):

        self.env = env
        self.max_memory_size = 10_000
        self.batch_size = 32
        self.actor_learning_rate = 1e-4

        self.memory = Memory(self.max_memory_size)

        self.hidden_size_network_model = [128, 64]
        self.num_states  = 3
        self.num_actions = 1

        self.model_transition_1 = ModelNet_transitions(self.num_states + self.num_actions, self.hidden_size_network_model, self.num_states)
        self.model_transition_2 = ModelNet_transitions(self.num_states + self.num_actions, self.hidden_size_network_model, self.num_states)
        self.model_transition_3 = ModelNet_transitions(self.num_states + self.num_actions, self.hidden_size_network_model, self.num_states)

        self.model_reward = ModelNet_reward(self.num_states + self.num_actions, self.hidden_size_network_model, 1)
        self.model_done   = ModelNet_N_done(self.num_states)

        self.model_transition_1_optimizer = optim.Adam(self.model_transition_1.parameters(), lr=self.actor_learning_rate)
        self.model_transition_2_optimizer = optim.Adam(self.model_transition_2.parameters(), lr=self.actor_learning_rate)
        self.model_transition_3_optimizer = optim.Adam(self.model_transition_3.parameters(), lr=self.actor_learning_rate)

        self.model_reward_optimizer = optim.Adam(self.model_reward.parameters(), lr=self.actor_learning_rate)



    def get_action_from_policy(self):
        # this policy be updated and learned
        action = self.env.action_space.sample()  # changes this
        return action

    def add_real_experience_memory(self, state, action, reward, next_state, done):
        self.memory.replay_buffer_environment_add(state, action, reward, next_state, done)

    def transition_model_learn(self):
        # "learning" the model....

        if self.memory.len_env_buffer() <= self.batch_size:
            return
        else:
            print("training the models.....")
            states, actions, rewards, next_states, dones = self.memory.sample_experience_from_env(self.batch_size)

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

            print(dones)

            # ---- Transition Model---- #
            prediction_transition_1 = self.model_transition_1.forward(states, actions)
            prediction_transition_2 = self.model_transition_2.forward(states, actions)
            prediction_transition_3 = self.model_transition_3.forward(states, actions)

            # ---- Reward Model ----- #
            predicted_reward = self.model_reward.forward(next_states, actions)

            # ---- Done Model ------#
            predicted_done = self.model_done.forward(next_states)




            '''
            # calculate the loss
            loss_function = nn.MSELoss(reduction='mean')

            model_loss_1 = loss_function(prediction_transition_1, next_states)
            self.model_transition_1_optimizer.zero_grad()
            model_loss_1.backward()
            self.model_transition_1_optimizer.step()

            model_loss_2 = loss_function(prediction_transition_2, next_states)
            self.model_transition_2_optimizer.zero_grad()
            model_loss_2.backward()
            self.model_transition_2_optimizer.step()

            reward_loss = loss_function(predicted_reward, rewards)
            self.model_reward_optimizer.zero_grad()
            reward_loss.backward()
            self.model_reward_optimizer.step()
            
            '''


    def generate_dream_samples(self):

        # todo think how many sample take for input? do i need just 1?
        M = 40
        K_steps = 2
        model_branches = 2
        number_samples = 4

        if self.memory.len_env_buffer() <= number_samples:
            return
        else:
            for forward in range(1, M):
                states, actions, rewards, next_states, dones = self.memory.sample_experience_from_env(number_samples)
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

                self.model_transition_1.eval()
                with torch.no_grad():
                    observation_generated = self.model_transition_1.forward(states, actions)
                    observation_generated = observation_generated.detach()
                    observation_generated = observation_generated.numpy()  # tensor to numpy
                    self.model_transition_1.train()

                self.model_reward.eval()
                with torch.no_grad():
                    reward_generated = self.model_reward.forward(states, actions)
                    reward_generated = reward_generated.detach()
                    reward_generated = reward_generated.numpy()  # tensor to numpy
                    self.model_reward.train()


                imaginative_experiences = 0


def main():

    env   = gym.make("Pendulum-v1")
    agent = ModelAgent(env)

    EPISODES = 10

    # todo add randome accion here for a considerable steps and add them to the  env memory buffer,
    #  totally random choices, exploration

    for episode in range(1, EPISODES + 1):

        print(f"-------Episode:{episode} ---------")

        state = env.reset()
        done = False


        F = 0
        while not done:

            action = agent.get_action_from_policy()

            next_state, reward, done, _ = env.step(action)

            agent.add_real_experience_memory(state, action, reward, next_state, done)

            state = next_state

            agent.transition_model_learn()

            #if F <= 50:
                #F += 1
                #agent.transition_model_learn()  # todo add some F values so fix number for training

            #agent.generate_dream_samples()

            if done:
                break






if __name__ == '__main__':
    main()
