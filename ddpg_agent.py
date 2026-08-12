
# Deep Deterministic Policy Gradient (DDPG) for continuous control.

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import time
from copy import deepcopy
import imageio.v2 as imageio
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
gif_dir = current_path + '/gifs/'
os.makedirs(model, exist_ok=True)
os.makedirs(ddpg_reward, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)
os.makedirs(gif_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d%H%M%S")

# Training hyperparameters
NUM_EPISODE = 150
NUM_STEP = 200
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000

# GIF recording uses a separate environment and a fixed evaluation seed.
GIF_FPS = 30
GIF_SEED = 123
GIF_EPISODES = {50, 100}

# Create the DDPG agent.
agent = DDPGAgent(STATE_DIM, ACTION_DIM)


def record_policy_gif(agent, output_path, seed=GIF_SEED, fps=GIF_FPS):
    """Record a deterministic evaluation episode as a GIF."""
    eval_env = gym.make(id='Pendulum-v1', render_mode='rgb_array')
    frames = []
    actor_was_training = agent.actor.training

    try:
        state, _ = eval_env.reset(seed=seed)
        frames.append(eval_env.render())

        # Record the deterministic policy without exploration noise.
        agent.actor.eval()
        with torch.no_grad():
            while True:
                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=next(agent.actor.parameters()).device,
                ).unsqueeze(0)
                action = agent.actor(state_tensor).cpu().numpy()[0]

                state, _, terminated, truncated, _ = eval_env.step(action)
                frames.append(eval_env.render())

                if terminated or truncated:
                    break
    finally:
        agent.actor.train(actor_was_training)
        eval_env.close()

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"Saved GIF: {output_path}")


# Record the untrained policy in a separate environment.
record_policy_gif(agent, os.path.join(gif_dir, 'before_training.gif'))

# Train the agent.
REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
best_average_reward = -float('inf')  # Track the best moving-average reward.
best_episode_reward = -float('inf')
best_actor_state = None

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
        next_state, reward, terminated, truncated, info = env.step(action)
        # Store only true terminations so time-limit truncations can bootstrap.
        agent.replay_buffer.add_memo(state, action, reward, next_state, terminated)

        state = next_state
        episode_reward += reward

        agent.update()
        if terminated or truncated:
            break

    # Track the actor from the highest-reward rollout before later updates.
    if episode_reward > best_episode_reward:
        best_episode_reward = episode_reward
        best_actor_state = deepcopy(agent.actor.state_dict())

    # GIF episode labels use one-based indices.
    episode_number = episode_i + 1
    if episode_number in GIF_EPISODES:
        record_policy_gif(
            agent,
            os.path.join(gif_dir, f'episode_{episode_number}.gif'),
        )

    REWARD_BUFFER[episode_i] = episode_reward
    Average_reward = np.mean(REWARD_BUFFER[max(0, episode_i - 100):(episode_i + 1)])
    if Average_reward >= -160 and Average_reward > best_average_reward:
        best_average_reward = Average_reward
        torch.save(agent.actor.state_dict(), model + f'ddpg_actor_{timestamp}_ar_{round(best_average_reward)}.pth')
    print(f"Episode: {episode_i}, Reward: {round(episode_reward, 2)}, Average Reward: {round(Average_reward, 2)}")

# Close the environment.
env.close()

# Record the best in-memory policy, then restore the final parameters.
if best_actor_state is not None:
    final_actor_state = deepcopy(agent.actor.state_dict())
    try:
        agent.actor.load_state_dict(best_actor_state)
        record_policy_gif(agent, os.path.join(gif_dir, 'best_policy.gif'))
    finally:
        agent.actor.load_state_dict(final_actor_state)

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

