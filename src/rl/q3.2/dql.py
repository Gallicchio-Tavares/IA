import numpy as np


class QLearningAgentTabularDiscrete:
    def __init__(self, env, bins=(20, 20), learning_rate=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995):
        self.env = env
        self.bins = bins
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros(bins + (env.action_space.n,))
        self.obs_space_low = env.observation_space.low
        self.obs_space_high = env.observation_space.high
        self.obs_space_bins = [np.linspace(self.obs_space_low[i], self.obs_space_high[i], self.bins[i]) for i in
                               range(len(self.bins))]

    def discretize(self, state):
        discretized_state = []
        for i in range(len(state)):
            discretized_state.append(np.digitize(state[i], self.obs_space_bins[i]) - 1)
        return tuple(discretized_state)

    def choose_action(self, state, is_in_exploration_mode=True):
        discretized_state = self.discretize(state)

        if is_in_exploration_mode and np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Exploração

        return np.argmax(self.q_table[discretized_state])  # Exploração


    def learn(self, state, action, reward, next_state, done):
        discretized_state = self.discretize(state)
        discretized_next_state = self.discretize(next_state)

        current_q_value = self.q_table[discretized_state + (action,)]
        max_future_q_value = np.max(self.q_table[discretized_next_state])

        new_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * (
                reward + self.gamma * max_future_q_value)
        self.q_table[discretized_state + (action,)] = new_q_value

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        np.save(filename, self.q_table)

    def load(self, filename):
        self.q_table = np.load(filename)
