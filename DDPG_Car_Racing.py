
# still workin here
import gym
from Noise import OUNoise
from Networks import CriticDDPG, Actor


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
        self.num_states = env.observation_space  # (96x96x3)
        self.num_actions = env.action_space.shape[0]   # 3






def main():
    EPISODES        = 10000  # Total number of episodes
    render_interval = EPISODES * 0.95  # Activate render after 95% of total episodes
    batch_size      = 64

    env   = gym.make('CarRacing-v1')
    agent = DDPGagent(env, batch_size=batch_size)
    noise = OUNoise(env.action_space)

    rewards     = []
    avg_rewards = []

    state = env.reset()
    print(state)


if __name__ == '__main__':
    main()
    #test_load()
