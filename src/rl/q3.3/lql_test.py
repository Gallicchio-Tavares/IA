import argparse
from lql import QLearningAgentLinear

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Blackjack-v1", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes")
    args = parser.parse_args()
    assert args.num_episodes > 0

    model_path = f"output/modelos/{args.env_name}-lql-agent.pkl"
    agent = QLearningAgentLinear.load_agent(model_path)

    total_actions, total_rewards = 0, 0

    for episode in range(args.num_episodes):

        state, _ = agent.env.reset()
        num_actions = 0
        reward = 0
        
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.policy(state)
            state, reward, terminated, truncated, info = agent.env.step(action)
            num_actions += 1

        total_rewards += reward
        total_actions += num_actions

    print("***Results***********************")
    print(f"Average episode length: {total_actions / args.num_episodes}")
    print(f"Average rewards: {total_rewards / args.num_episodes}")
    print("**********************************")
