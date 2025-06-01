import argparse
from tql import QLearningAgentTabular

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Blackjack-v1", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes")
    args = parser.parse_args()
    assert args.num_episodes > 0

    model_path = f"output/modelos/{args.env_name}-tql-agent.pkl"
    agent = QLearningAgentTabular.load_agent(model_path)

    total_actions = 0
    total_rewards = 0
    successful_episodes = 0

    for episode in range(args.num_episodes):
        state, _ = agent.env.reset()
        state_id = agent.env.get_state_id(state)

        num_actions = 0
        episode_reward = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.choose_action(state_id, is_in_exploration_mode=False)  # modo greedy
            new_state, reward, terminated, truncated, info = agent.env.step(action)
            new_state_id = agent.env.get_state_id(new_state)

            state = new_state
            state_id = new_state_id

            episode_reward += reward
            num_actions += 1

        total_rewards += episode_reward
        total_actions += num_actions

        # Sucesso depende do ambiente:
        if reward > 0:
            successful_episodes += 1

    avg_episode_length = total_actions / args.num_episodes
    avg_rewards = total_rewards / args.num_episodes
    success_rate = (successful_episodes / args.num_episodes) * 100

    print("\n*** Results **************************")
    print(f"Environment: {args.env_name}")
    print(f"Average episode length: {avg_episode_length:.2f}")
    print(f"Average rewards: {avg_rewards:.2f}")
    print(f"Success rate (reward > 0): {success_rate:.2f}%")
    print("****************************************\n")
