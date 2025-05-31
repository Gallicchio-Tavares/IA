import gymnasium as gym
import numpy as np


class AbstractMountainCarEnvironment:
    def get_num_actions(self):
        raise NotImplementedError

    def get_num_states(self):
        raise NotImplementedError

    def get_random_action(self):
        raise NotImplementedError

    def get_state_id(self, state):
        raise NotImplementedError


class MountainCarEnvironment(AbstractMountainCarEnvironment):
    def __init__(self, bins=(20, 20)):
        self.env = gym.make("MountainCar-v0")
        self.bins = bins
        self.obs_space_low = self.env.observation_space.low
        self.obs_space_high = self.env.observation_space.high
        self.obs_space_bins = [np.linspace(self.obs_space_low[i], self.obs_space_high[i], bins[i]) for i in
                               range(len(bins))]
        self.action_space_n = self.env.action_space.n

    def discretize(self, state):
        discretized_state = []
        for i in range(len(state)):
            discretized_state.append(np.digitize(state[i], self.obs_space_bins[i]) - 1)
        return tuple(discretized_state)

    def get_num_actions(self):
        return self.action_space_n

    def get_num_states(self):
        return self.bins + (self.action_space_n,)

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_state_id(self, state):
        return state
