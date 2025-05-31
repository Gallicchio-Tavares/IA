import gymnasium as gym
import numpy as np
from environments.environment import Environment


class GymEnvironment(Environment):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.env.action_space, gym.spaces.Discrete)

    def get_num_states(self):
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            return self.env.observation_space.n
        else:
            raise NotImplementedError("Observação não suportada para este wrapper.")

    def get_num_actions(self):
        return self.env.action_space.n

    def get_state_id(self, observation):
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            return observation
        else:
            raise NotImplementedError("Observação não suportada para este wrapper.")

    def get_random_action(self):
        return self.env.action_space.sample()
