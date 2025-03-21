import gymnasium as gym
import torch
import numpy as np
from pacman import DuelingQNetwork, stack_frames, device, N_ATOM  # Import necessary classes and variables
from gymnasium.wrappers import AtariPreprocessing
import matplotlib.pyplot as plt


def test_agent(checkpoint_path, episodes=10):
    env = gym.make("ALE/MsPacman-v5", render_mode="human", frameskip=1)
    env = AtariPreprocessing(env)
    n_actions = env.action_space.n
    state_shape = (4, 84, 84)

    agent = DuelingQNetwork(state_shape, n_actions, N_ATOM).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()  # Set the agent to evaluation mode

    all_rewards = []

    for episode in range(episodes):
        state, info = env.reset()
        stacked_frames, stacked_state = stack_frames(None, state, True)
        total_reward = 0
        done = False

        while not done:
            # Choose action using the loaded agent
            state_tensor = torch.from_numpy(stacked_state).float().unsqueeze(0).to(device)
            with torch.no_grad():  # No need to calculate gradients during testing
                q_values = agent.get_q_values(state_tensor)
            action = np.argmax(q_values.cpu().data.numpy())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            stacked_frames, stacked_next_state = stack_frames(stacked_frames, next_state, False)
            stacked_state = stacked_next_state
            total_reward += reward

        all_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()
    print(f"Average reward over {episodes} episodes: {np.mean(all_rewards)}")

    # Plotting
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Agent Performance over Episodes")
    plt.show()


if __name__ == "__main__":
    checkpoint_file = "checkpoint_10.pth"  # Replace with the actual path to your checkpoint file
    test_agent(checkpoint_file, episodes=10)