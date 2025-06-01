import argparse
import gymnasium as gym
from lql import QLearningAgentLinear
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from gymnasium.wrappers import TimeLimit
from environments.cliffwalking_environment import CliffWalkingEnvironment
from environments.frozenlake_environment import FrozenLakeEnvironment

#from taxi_environment import TaxiEnvironment
from environments.blackjack_environment import BlackjackEnvironment

environment_dict = {
    "Blackjack-v1": BlackjackEnvironment,
    "CliffWalking-v0": CliffWalkingEnvironment,
    "FrozenLake-v1": FrozenLakeEnvironment,
    #"Taxi-v3": TaxiEnvironment
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=10000, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum number of steps per training episode")
    parser.add_argument("--env_name", type=str, default="Blackjack-v1", help="Environment name")
    parser.add_argument("--epsilon_decay_rate", type=float, default=0.0001, help="Decay rate for the exploration rate")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Gamma")
    args = parser.parse_args()

    num_episodes = args.num_episodes
    max_steps = args.max_steps
    env_name = args.env_name
    epsilon_decay_rate = args.epsilon_decay_rate
    learning_rate = args.learning_rate
    gamma = args.gamma

    env = gym.make(env_name)
    env = TimeLimit(env, max_episode_steps=args.max_steps)
    env = environment_dict[env_name](env)

    agent = QLearningAgentLinear(env, learning_rate=learning_rate, epsilon_decay_rate=epsilon_decay_rate, gamma=gamma)
    penalties_per_episode, rewards_per_episode, cumulative_successful_episodes = agent.train(num_episodes)
    agent.save(args.env_name + "-lql-agent.pkl")

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(savgol_filter(penalties_per_episode, 51, 2), color='red')
plt.title(f"Penalties per Episode\n({args.env_name})", fontsize=10)
plt.xlabel("Episode")
plt.ylabel("Penalties")
plt.grid(alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(savgol_filter(rewards_per_episode, 51, 2), color='green')
plt.title(f"Rewards per Episode\n({args.env_name})", fontsize=10)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(cumulative_successful_episodes, color='blue')
plt.title(f"Cumulative Successful Episodes\n({args.env_name})", fontsize=10)
plt.xlabel("Episode")
plt.ylabel("Success Count")
plt.grid(alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(agent.epsilon_history, color='purple')
plt.title(f"Epsilon Decay\n({args.env_name})", fontsize=10)
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{args.env_name}-lql-results.png", dpi=300)
plt.close()
