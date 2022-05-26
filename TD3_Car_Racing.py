import gym
from TD3_NN import Actor, Critic
import torch


class TD3(object):
    def __init__(self, action_dim, img_stack):

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
        state = state.float()
        return self.actor(state).cpu().data.numpy().flatten()


def main():
    env = gym.make('CarRacing-v0')
    action_dim = env.action_space.shape[0]
    state = env.reset()
    img_stack = 4  # number of image stacks together
    TD3(action_dim, img_stack)



if __name__ == '__main__':
    main()
