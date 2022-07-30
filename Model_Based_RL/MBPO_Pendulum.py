# Based on When to Trust Your Model: Model-Based Policy Optimization

import gym
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
import matplotlib.pyplot as plt


class Memory:

    def __init__(self, replay_max_size_env, replay_max_size_model):

        self.replay_max_size_env   = replay_max_size_env
        self.replay_max_size_model = replay_max_size_model

        self.replay_buffer_env   = deque(maxlen=replay_max_size_env)
        self.replay_buffer_model = deque(maxlen=replay_max_size_model)

    def replay_buffer_environment_add(self, state, action, reward, next_state, done):
        experience_from_env = (state, action, reward, next_state, done)
        self.replay_buffer_env.append(experience_from_env)

    def replay_buffer_model_add(self, state, action, reward, next_state, done):
        experience_from_model = (state, action, reward, next_state, done)
        self.replay_buffer_model.append(experience_from_model)

    def sample_experience_from_env(self, batch_size):
        state_batch  = []
        action_batch = []
        reward_batch = []
        done_batch   = []
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


    def sample_experience_from_model(self, batch_size):
        state_batch  = []
        action_batch = []
        reward_batch = []
        done_batch   = []
        next_state_batch = []

        batch_of_experiences = random.sample(self.replay_buffer_model, batch_size)

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


class ModelNet_probabilistic_transition(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(ModelNet_probabilistic_transition, self).__init__()

        self.mean_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size[0], bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size[0], hidden_size[1], bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size[1], hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size[2], 1)
        )

        self.std_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size[0], bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size[0], hidden_size[1], bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size[1], hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size[2], 1),
            nn.Softplus()
        )

    def forward(self, state, action):
        
        x   = torch.cat([state, action], dim=1)  # Concatenates the seq tensors in the given dimension
        u   = self.mean_layer(x)
        std = self.std_layer(x)
        
        return torch.distributions.Normal(u, std)


class ModelNet_transitions(nn.Module):
    # This neural network uses action and current state as inputs.
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNet_transitions, self).__init__()
        self.h_linear_1 = nn.Linear(in_features=input_size, out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=output_size)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = self.h_linear_3(x)
        return x


class ModelNet_probabilistic_reward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNet_probabilistic_reward, self).__init__()
        self.mean_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size[0], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[1], hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[2], 1)
        )
        self.std_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size[0], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[1], hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[2], 1),
            nn.Softplus()
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenates the seq tensors in the given dimension
        u = self.mean_layer(x)
        std = self.std_layer(x)
        return torch.distributions.Normal(u, std)



class ModelNet_reward(nn.Module):
    # This neural network uses action and next state as inputs.
    def __init__(self, input_size, hidden_size):
        super(ModelNet_reward, self).__init__()
        self.reward_predicted_model = nn.Sequential(
            nn.Linear(input_size, hidden_size[0], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[1], hidden_size[2], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[2], 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenates the seq tensors in the given dimension
        r = self.reward_predicted_model(x)
        return r


class ModelNet_N_done(nn.Module):
    #  This neural network uses only next state as inputs.
    def __init__(self, input_size):
        super(ModelNet_N_done, self).__init__()
        hidden_size = [256, 256, 256]
        self.h_linear_1 = nn.Linear(in_features=input_size, out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=1)

    def forward(self, next_state):
        x = next_state
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = torch.sigmoid(self.h_linear_4(x))
        return x


class CriticTD3(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(CriticTD3, self).__init__()

        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=num_actions)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)   # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = self.h_linear_4(x)                  # No activation function here
        return x


class ActorTD3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorTD3, self).__init__()

        self.h_linear_1 = nn.Linear(in_features=input_size,     out_features=hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2])
        self.bn1 = nn.BatchNorm1d(hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=hidden_size[2], out_features=output_size)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.bn1(self.h_linear_3(x)))
        x = torch.tanh(self.h_linear_4(x))
        return x


class ModelAgent:
    def __init__(self, env):
        # -------- Hyper-parameters --------------- #
        self.env = env
        self.batch_size = 64
        self.max_memory_size_env   = 50_000
        self.max_memory_size_model = 50_000

        self.reward_learning_rate     = 1e-2
        self.transition_learning_rate = 1e-3

        self.actor_learning_rate  = 1e-4
        self.critic_learning_rate = 1e-3

        self.hidden_size_network_model  = [128, 128, 64, 32]
        self.hidden_size_network_reward = [64, 32, 32]

        self.hidden_size_critic = [512, 512, 256]
        self.hidden_size_actor  = [256, 256, 256]

        self.num_states  = 3
        self.num_actions = 1

        self.loss_model_1 = []
        self.loss_reward  = []

        self.gamma = 0.99
        self.tau   = 0.005
        self.inter_act_counter  = 0
        self.update_counter     = 0
        self.policy_freq_update = 2

        self.tolerance_loss = 1e-5

        # ------------- Initialization memory --------------------- #
        self.memory = Memory(self.max_memory_size_env, self.max_memory_size_model)

        # ---------- Initialization and build the networks for TD3 ----------- #
        # Main networks
        self.actor     = ActorTD3(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic_q1 = CriticTD3(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)
        self.critic_q2 = CriticTD3(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

        # Target networks
        self.actor_target     = ActorTD3(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic_target_q1 = CriticTD3(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)
        self.critic_target_q2 = CriticTD3(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_q1.parameters(), self.critic_q1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_q2.parameters(), self.critic_q2.parameters()):
            target_param.data.copy_(param.data)


        # ---------- Initialization and build the networks for Model Learning ----------- #
        # Deterministic
        self.model_transition_1 = ModelNet_transitions(self.num_states + self.num_actions,
                                                       self.hidden_size_network_model, self.num_states)
        self.model_transition_2 = ModelNet_transitions(self.num_states + self.num_actions,
                                                       self.hidden_size_network_model, self.num_states)
        self.model_transition_3 = ModelNet_transitions(self.num_states + self.num_actions,
                                                       self.hidden_size_network_model, self.num_states)

        # Reward
        self.model_reward = ModelNet_reward(self.num_states + self.num_actions, self.hidden_size_network_reward)

        # ----------------------------------------------------------------------------------------------
        # Probabilistic
        self.pdf_transition_1 = ModelNet_probabilistic_transition(self.num_states + self.num_actions,
                                                                  self.hidden_size_network_model)
        self.pdf_transition_2 = ModelNet_probabilistic_transition(self.num_states + self.num_actions,
                                                                  self.hidden_size_network_model)
        self.pdf_transition_3 = ModelNet_probabilistic_transition(self.num_states + self.num_actions,
                                                                  self.hidden_size_network_model)

        # Reward
        self.pdf_reward = ModelNet_probabilistic_reward(self.num_states + self.num_actions,
                                                                  self.hidden_size_network_reward)

        # ------------------------------------------------------------------------------------------------
        self.model_transition_1_optimizer = optim.Adam(self.model_transition_1.parameters(),
                                                       lr=self.transition_learning_rate)
        self.model_transition_2_optimizer = optim.Adam(self.model_transition_2.parameters(),
                                                       lr=self.transition_learning_rate)
        self.model_transition_3_optimizer = optim.Adam(self.model_transition_3.parameters(),
                                                       lr=self.transition_learning_rate)

        self.pdf_transition_1_optimizer = optim.Adam(self.pdf_transition_1.parameters(),
                                                     lr=self.transition_learning_rate, weight_decay=1e-5)
        self.pdf_transition_2_optimizer = optim.Adam(self.pdf_transition_2.parameters(),
                                                     lr=self.transition_learning_rate, weight_decay=1e-5)
        self.pdf_transition_3_optimizer = optim.Adam(self.pdf_transition_3.parameters(),
                                                     lr=self.transition_learning_rate, weight_decay=1e-5)

        self.model_reward_optimizer = optim.Adam(self.model_reward.parameters(), lr=self.reward_learning_rate)

        self.actor_optimizer    = optim.Adam(self.actor.parameters(),     lr=self.actor_learning_rate)
        self.critic_optimizer_1 = optim.Adam(self.critic_q1.parameters(), lr=self.critic_learning_rate)
        self.critic_optimizer_2 = optim.Adam(self.critic_q2.parameters(), lr=self.critic_learning_rate)


    def get_action_from_policy(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.forward(state_tensor)
            action = action.detach()
            action = action.numpy()  # tensor to numpy
            self.actor.train()
        return action[0]

    def add_real_experience_memory(self, state, action, reward, next_state, done):
        self.memory.replay_buffer_environment_add(state, action, reward, next_state, done)

    def add_imagined_experience_memory(self, state, action, reward, next_state, done):
        self.memory.replay_buffer_model_add(state, action, reward, next_state, done)

    def transition_model_learn(self, mode):

        # learning" the model....

        if self.memory.len_env_buffer() <= self.batch_size:
            return
        else:
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

            if mode == "Deterministic":
                # make the prediction
                delta_prediction      = self.model_transition_1.forward(states, actions)
                next_states_predicted = states + delta_prediction
                predicted_reward      = self.model_reward.forward(states, actions)

                # calculate the loss

                # Transitions Model ----- #
                loss_fn = torch.nn.MSELoss(reduction='mean')
                prediction_loss = loss_fn(next_states_predicted, next_states)

                self.model_transition_1_optimizer.zero_grad()
                prediction_loss.backward()
                self.model_transition_1_optimizer.step()
                self.loss_model_1.append(prediction_loss.item())

                # Reward Model ----- #
                loss_reward = loss_fn(predicted_reward, rewards)
                self.model_reward_optimizer.zero_grad()
                loss_reward.backward()
                self.model_reward_optimizer.step()
                self.loss_reward.append(loss_reward.item())

            elif mode == "Probabilistic":

                # ---- Transition Model---- #
                distribution_probability_model_1 = self.pdf_transition_1.forward(states, actions)
                distribution_probability_model_2 = self.pdf_transition_2.forward(states, actions)
                distribution_probability_model_3 = self.pdf_transition_3.forward(states, actions)

                distribution_probability_reward = self.pdf_reward.forward(states, actions)

                # calculate the loss
                loss_neg_log_likelihood_1 = - distribution_probability_model_1.log_prob(next_states)
                loss_neg_log_likelihood_1 = torch.mean(loss_neg_log_likelihood_1)

                loss_neg_log_likelihood_2 = - distribution_probability_model_2.log_prob(next_states)
                loss_neg_log_likelihood_2 = torch.mean(loss_neg_log_likelihood_2)

                loss_neg_log_likelihood_3 = - distribution_probability_model_3.log_prob(next_states)
                loss_neg_log_likelihood_3 = torch.mean(loss_neg_log_likelihood_3)

                loss_neg_loss_likelihood_reward = -distribution_probability_reward.log_prob(rewards)
                loss_neg_loss_likelihood_reward = torch.mean(loss_neg_loss_likelihood_reward)

                
                self.pdf_transition_1.train()
                self.pdf_transition_1_optimizer.zero_grad()
                loss_neg_log_likelihood_1.backward()
                self.pdf_transition_1_optimizer.step()

                self.pdf_transition_2_optimizer.zero_grad()
                loss_neg_log_likelihood_2.backward()
                self.pdf_transition_2_optimizer.step()

                self.pdf_transition_3_optimizer.zero_grad()
                loss_neg_log_likelihood_3.backward()
                self.pdf_transition_3_optimizer.step()

                self.model_reward_optimizer.zero_grad()
                loss_neg_loss_likelihood_reward.backward()
                self.model_reward_optimizer.step()

                self.loss_model_1.append(loss_neg_log_likelihood_1.item())
                self.loss_reward.append(loss_neg_loss_likelihood_reward.item())

                print(loss_neg_log_likelihood_1.item())


    def generate_dream_samples(self, mode):
        M = 5
        K_steps = 1
        number_samples = 1  # I need to take a single sample always, fixed value
        decision = "random"

        if self.memory.len_env_buffer() <= number_samples:
            return
        else:
            for _ in range(1, M+1):
                state, _, _, _, _ = self.memory.sample_experience_from_env(number_samples)
                state = np.array(state)  # to get an array of (1,3)
                state_tensor = torch.FloatTensor(state)  # torch.Size([1, 3])

                # THE ACTION COMES FROM THE AGENT AND POLICY
                action = self.get_action_from_policy(state[0])  # array  of (1,)
                noise  = (np.random.normal(0, scale=0.1, size=1))  # todo have to put this in here?
                action = action + noise
                action = np.clip(action, -1, 1)

                action_tensor = torch.FloatTensor(action)
                action_tensor = action_tensor.unsqueeze(0)  # torch.Size([1, 1])

                if mode == "Deterministic":
                    self.model_transition_1.eval()
                    with torch.no_grad():
                        imagined_prediction = self.model_transition_1.forward(state_tensor, action_tensor)  # torch.Size([1, 3])
                        imagined_prediction = imagined_prediction.detach()
                        imagined_prediction = imagined_prediction.numpy()
                        self.model_transition_1.train()

                    self.model_reward.eval()
                    with torch.no_grad():
                        reward_generated = self.model_reward.forward(state_tensor, action_tensor)
                        reward_generated = reward_generated.detach()
                        reward_generated = reward_generated.numpy()  # tensor to numpy
                        self.model_reward.train()

                    done = False
                    self.add_imagined_experience_memory(state[0], action, reward_generated[0], imagined_prediction[0], done)

                '''
                self.pdf_transition_1.eval()
                with torch.no_grad():
                    function_generated_1    = self.pdf_transition_1.forward(state_tensor, action_tensor)
                    observation_generated_1 = function_generated_1.sample((3,))  # torch.Size([3, 1, 1])
                    observation_generated_1 = torch.reshape(observation_generated_1, (1, 3))  # torch.Size([1, 3])
                    observation_generated_1 = observation_generated_1.detach()
                    observation_generated_1 = observation_generated_1.numpy()
                    self.pdf_transition_1.train()

                self.pdf_transition_2.eval()
                with torch.no_grad():
                    function_generated_2 = self.pdf_transition_2.forward(state_tensor, action_tensor)
                    observation_generated_2 = function_generated_2.sample((3,))  # torch.Size([3, 1, 1])
                    observation_generated_2 = torch.reshape(observation_generated_2, (1, 3))  # torch.Size([1, 3])
                    observation_generated_2 = observation_generated_2.detach()
                    observation_generated_2 = observation_generated_2.numpy()
                    self.pdf_transition_2.train()

                self.pdf_transition_3.eval()
                with torch.no_grad():
                    function_generated_3 = self.pdf_transition_3.forward(state_tensor, action_tensor)
                    observation_generated_3 = function_generated_1.sample((3,))  # torch.Size([3, 1, 1])
                    observation_generated_3 = torch.reshape(observation_generated_3, (1, 3))  # torch.Size([1, 3])
                    observation_generated_3 = observation_generated_3.detach()
                    observation_generated_3 = observation_generated_3.numpy()
                    self.pdf_transition_3.train()

                self.model_reward.eval()
                with torch.no_grad():
                    reward_generated = self.model_reward.forward(state_tensor, action_tensor)
                    reward_generated = reward_generated.detach()
                    reward_generated = reward_generated.numpy()  # tensor to numpy
                    self.model_reward.train()

                # todo the reward could be probabilistic too?
                # todo for done values calculate using a function based on the prediction, reward and action
                if decision == "random":
                    model_choose = random.randint(1, 3)
                    if model_choose == 1:
                        next_state_imagined = observation_generated_1
                    elif model_choose == 2:
                        next_state_imagined = observation_generated_2
                    elif model_choose == 3:
                        next_state_imagined = observation_generated_3
                else:
                    # todo a single value between all the prediction, a average/mean value
                    pass

                done = False
                # todo fix done value here

                self.add_imagined_experience_memory(state[0], action, reward_generated[0], next_state_imagined[0], done)
                '''

    def step_training(self):

        if self.memory.len_env_buffer() and self.memory.len_model_buffer() <= self.batch_size:
            return
        else:
            G = 6
            for it in range(1, G+1):

                self.update_counter += 1  # todo carefully where to put this number

                states, actions, rewards, next_states, dones = self.memory.sample_experience_from_model(self.batch_size)

                states  = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards).reshape(-1, 1)
                dones = np.array(dones).reshape(-1, 1)
                next_states = np.array(next_states)

                states  = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards)
                dones = torch.FloatTensor(dones)
                next_states = torch.FloatTensor(next_states)

                # ------- compute the target action
                next_actions = self.actor_target.forward(next_states)

                # add noise also here, paper mention this
                next_actions = next_actions.detach().numpy()  # tensor to numpy
                next_actions = next_actions + (np.random.normal(0, scale=0.2, size=self.num_actions))
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


def main():

    env = gym.make("Pendulum-v1")
    agent = ModelAgent(env)

    EPISODES = 100
    Initial_Exploration_episodes = 50
    F = 50

    rewards     = []
    avg_rewards = []

    #mode = "Deterministic"
    mode = "Probabilistic"

    for explo_step in range(1, Initial_Exploration_episodes + 1):
        state = env.reset()
        done = False
        print("exploration:", explo_step)

        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            agent.add_real_experience_memory(state, action, reward, next_state, done)
            if done:
                break

    for episode in range(1, EPISODES + 1):
        print(f"-------Episode:{episode} ---------")
        state = env.reset()
        done  = False
        episode_reward = 0
        step = 0

        while not done:

            step += 1

            #action = agent.get_action_from_policy(state)
            action = env.action_space.sample()
            #noise  = (np.random.normal(0, scale=0.1, size=1))
            #action = action + noise
            action = np.clip(action, -1, 1)
            convert_action = (action - (-1)) * (2 - (-2)) / (1 - (-1)) + (-2)  # to put the output between +2 -2 # todo Do I need this?
            next_state, reward, done, _ = env.step(convert_action)

            agent.add_real_experience_memory(state, action, reward, next_state, done)
            state = next_state

            #if step <= F:
            agent.transition_model_learn(mode=mode)

            #agent.generate_dream_samples(mode=mode)
            #agent.step_training()

            episode_reward += reward
            if done:
                break


        print(f"******* -----Episode {episode} Ended-----********* ")
        print("Episode total reward:", episode_reward)
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-100:]))


    plt.ylabel('Model Loss')
    plt.xlabel('steps')
    #plt.title('Training Curve Prediction')
    plt.plot(np.array(agent.loss_model_1))
    #plt.plot(np.array(agent.loss_reward))
    plt.legend(["transition", "Reward"])
    plt.show()

    #plt.plot(rewards)
    #plt.plot(avg_rewards)
    #plt.plot()
    #plt.title("MBPO-TD3")
    #plt.xlabel('Episode')
    #plt.ylabel('Reward')
    #plt.show()



if __name__ == '__main__':
    main()
