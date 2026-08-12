
# Deep Deterministic Policy Gradient (DDPG) for continuous control.

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import time
from ddpg_agent_notes import DDPGAgent
import random

# Create the training environment.
env = gym.make(id='Pendulum-v1')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
ddpg_reward = current_path + '/reward/'
figure_dir = current_path + '/figures/'
os.makedirs(model, exist_ok=True)
os.makedirs(ddpg_reward, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d%H%M%S")

# Training hyperparameters
NUM_EPISODE = 150
NUM_STEP = 200
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000

# Create the DDPG agent.
agent = DDPGAgent(STATE_DIM, ACTION_DIM)

# Train the agent.
REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
best_average_reward = -float('inf')  # Track the best moving-average reward.

for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    episode_reward = 0

    for step_i in range(NUM_STEP):
        # Select an exploratory or policy action.
        epsilon = np.interp(x=episode_i * NUM_STEP + step_i, xp=[0, EPSILON_DECAY],
                            fp=[EPSILON_START, EPSILON_END])  # Linearly decay exploration probability.
        random_sample = random.random()
        if random_sample <= epsilon:
            action = np.random.uniform(low=-2, high=2, size=ACTION_DIM)
        else:
            action = agent.get_action(state)
        # Step the environment.
        next_state, reward, done, truncation, info = env.step(action)
        # Store the transition.
        agent.replay_buffer.add_memo(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        agent.update()
        if done:
            break

    REWARD_BUFFER[episode_i] = episode_reward
    Average_reward = np.mean(REWARD_BUFFER[max(0, episode_i - 100):(episode_i + 1)])
    if Average_reward >= -180 and Average_reward > best_average_reward:
        best_average_reward = Average_reward
        torch.save(agent.actor.state_dict(), model + f'ddpg_actor_{timestamp}_ar_{round(best_average_reward)}.pth')
    print(f"Episode: {episode_i}, Reward: {round(episode_reward, 2)}, Average Reward: {round(Average_reward, 2)}")

# Close the environment.
env.close()

# Save rewards.
np.savetxt(ddpg_reward + f'/ddpg_reward_{timestamp}.txt', REWARD_BUFFER)

# Plot training rewards.
episodes = np.arange(NUM_EPISODE)
plt.figure(figsize=(10, 6))
plt.plot(episodes, REWARD_BUFFER, color='blue', alpha=0.5, label='Reward per Episode')
plt.plot(episodes, gaussian_filter1d(REWARD_BUFFER, sigma=5), color='blue', linewidth=2, label='Smoothed Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DDPG Reward')
plt.grid()
plt.legend()
plt.savefig(figure_dir + f"Rewards-{timestamp}.png", format='png', dpi=300)
plt.show()

