"""
Description:
			Aim: approximate the transition function of the environment


"""

import gym
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

import matplotlib.pyplot as plt

learning_rate = 1e-2

dtype = torch.float
device = torch.device("cpu")


def collect_data_vector_txt():
    env = gym.make('Pendulum-v1')
    filename = 'data_collected/data_vector.txt'
    outfile = open(filename, 'w')

    EPISODES = 5000  # Total number of episodes

    for episode in range(1, EPISODES + 1):
        print(f"-------Episode:{episode}---------")
        state = env.reset()
        done = False

        while not done:
            # env.render()
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            outfile.write("%s, %s, %s, %s, %s\n" % (state, action, reward, next_state, done))
            state = next_state
            if done:
                break


def collect_data_vector():
    env = gym.make('Pendulum-v1')

    actions_v = []
    states_v = []
    next_state_v = []
    reward_v = []

    for i_episode in range(1, 500):
        print("Episode:", i_episode)

        state = env.reset()
        done = False

        while not done:

            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            actions_v.append(action)
            states_v.append(state)
            next_state_v.append(next_state)
            reward_v.append(reward)

            state = next_state
            if done:
                break

    train_data = np.array([states_v, actions_v, next_state_v, reward_v])

    np.save("data_collected/train_data", train_data)


def model_learn():
    train_data = np.vstack(np.load('data_collected/train_data.npy', allow_pickle=True))

    state = np.vstack(train_data[0]).astype(float)
    action = np.vstack(train_data[1]).astype(float)
    next_state = np.vstack(train_data[2]).astype(float)
    reward = np.vstack(train_data[3]).astype(float)

    x_train = np.concatenate((state, action), axis=1)
    # y_train = next_state
    y_train = reward

    x = torch.tensor(x_train, requires_grad=True, device=device, dtype=dtype)
    y = torch.tensor(y_train, requires_grad=True, device=device, dtype=dtype)

    hidden_size = [64, 32, 32]

    model = torch.nn.Sequential(torch.nn.Linear(4, hidden_size[0]),
                                torch.nn.ReLU(),
                                torch.nn.Linear(hidden_size[0], hidden_size[1], bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Linear(hidden_size[1], hidden_size[2], bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Linear(hidden_size[2], 1, bias=True)).to(device)

    # model = torch.nn.Sequential(torch.nn.Linear(4, 128, bias=True),
    # torch.nn.ReLU(),
    # torch.nn.Linear(128, 3, bias=True),
    # ).to(device)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainloader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=False)

    tol = 1e-3
    running_loss = 100.0

    hold_loss = []

    for epoch in range(100):

        if running_loss < tol:
            break

        batch_iter = 0

        for x_batch, y_batch in trainloader:

            batch_iter += 1

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)

            running_loss = loss.item()

            print('epoch ({},{}), loss {}'.format(epoch, batch_iter, loss.item()))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            hold_loss.append(running_loss)

            if running_loss < tol:
                print("loss converged")
                break

    torch.save(model.state_dict(), 'pendulum_model.pt')

    plt.ylabel('MSE Loss')
    plt.xlabel('epochs')
    plt.title('Training Curve')

    plt.plot(np.array(hold_loss))
    plt.show()


def main():
    # collect_data_vector()
    model_learn()


if __name__ == '__main__':
    main()
