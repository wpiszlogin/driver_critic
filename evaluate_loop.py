"""
Main Loop of simulation.
Here a problem and a solution are defined.
"""

import gym
from pyglet.window import key
import numpy as np

from tools import *
from base_solution import *


# Parameters
n_episodes = 5000
problem = 'CarRacing-v0'

gym.logger.set_level(40)
all_episode_reward = []

# Initialize simulation
env = gym.make(problem)
env.reset()

# Define custom standard deviation for noise
# We can improve stability of solution, by noise parameters
noise_mean = np.array([0.0, -0.83], dtype=np.float32)
noise_std = np.array([0.0, 4 * 0.02], dtype=np.float32)
solution = BaseSolution(env.action_space, model_outputs=2, noise_mean=noise_mean, noise_std=noise_std)
solution.load_solution('models/best_solution/')


# Loop of episodes
for ie in range(n_episodes):
    state = env.reset()
    solution.reset()
    done = False
    episode_reward = 0
    no_reward_counter = 0

    # One-step-loop
    while not done:
        env.render()

        action, train_action = solution.get_action(state, add_noise=True)

        # This will make steering much easier
        action /= 4
        new_state, reward, done, info = env.step(action)

        state = new_state
        episode_reward += reward

        if reward < 0:
            no_reward_counter += 1
            if no_reward_counter > 200:
                break
        else:
            no_reward_counter = 0

    all_episode_reward.append(episode_reward)
    average_result = np.array(all_episode_reward[-100:]).mean()
    print('Last result:', episode_reward, 'Average results:', average_result)
