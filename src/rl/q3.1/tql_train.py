from timeit import default_timer as timer
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from tql import QLearningAgentTabular
from environments.gym_environment import GymEnvironment
from environments.blackjack_environment import BlackjackEnvironment

import os

os.makedirs("output/modelos", exist_ok=True)
os.makedirs("output/graficos", exist_ok=True)

environment_dict = {
    "Blackjack-v1": BlackjackEnvironment,
    "CliffWalking-v0": GymEnvironment,
    "FrozenLake-v1": GymEnvironment
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Blackjack-v1", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=6000, help="Number of episodes")
    parser.add_argument("--decay_rate", type=float, default=0.0001, help="Decay rate")
    parser.add_argument("--learning_rate", type=float, default=0.7, help="Learning rate (alpha)")
    parser.add_argument("--gamma", type=float, default=0.618, help="Discount factor (gamma)")
    args = parser.parse_args()

    num_episodes = args.num_episodes
    env_name = args.env_name
    decay_rate = args.decay_rate
    learning_rate = args.learning_rate
    gamma = args.gamma

    env = gym.make(env_name).env
    env = environment_dict[env_name](env)

    agent = QLearningAgentTabular(
        env=env,
        decay_rate=decay_rate,
        learning_rate=learning_rate,
        gamma=gamma
    )

    rewards = agent.train(num_episodes)

    agent.save(f"output/modelos/{env_name}-tql-agent.pkl")

    plt.plot(savgol_filter(rewards, 1001, 2))
    plt.title(f"Curva de aprendizado suavizada ({env_name})")
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa total')
    plt.savefig(f"output/graficos/{env_name}-tql-learning_curve.png")
    plt.close()

    plt.plot(agent.epsilons_)
    plt.title(f"Decaimento do valor de ε ({env_name})")
    plt.xlabel('Episódio')
    plt.ylabel('ε')
    plt.savefig(f"output/graficos/{env_name}-tql-epsilons.png")
    plt.close()

    print(f"Treinamento em {env_name} concluído. Modelos e gráficos salvos na pasta output/.")
