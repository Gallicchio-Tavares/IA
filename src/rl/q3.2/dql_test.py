import argparse
import gymnasium as gym
from dql import QLearningAgentTabularDiscrete

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="MountainCar-v0", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes")
    args = parser.parse_args()

    # Inicializa o ambiente
    env = gym.make(args.env_name)
    #env = gym.make(args.env_name, render_mode="human").env

    # Inicializa o agente com o ambiente e carrega o modelo treinado
    agent = QLearningAgentTabularDiscrete(env)
    agent.load("dql_mountaincar.npy")

    agent.epilon = 0

    total_rewards = 0
    total_actions = 0

    # Executa os episódios de teste
    for episode in range(args.num_episodes):
        state, _ = env.env.reset()  # Chamar o método reset no ambiente real
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.env.step(action)  # Chamar step no ambiente real
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

    print("***Results***********************")
    print(f"Average episode length: {total_actions / args.num_episodes}")
    print(f"Average rewards: {total_rewards / args.num_episodes}")
    print("**********************************")
