"""
Main Loop of simulation.
Here a problem and a solution are defined.
"""

import gym
from pyglet.window import key
import numpy as np

from tools import *
from base_solution import *


# Show preview
def key_press(k, mod):
    if k == key.SPACE:
        global preview
        preview = True


def key_release(k, mod):
    if k == key.SPACE:
        global preview
        preview = False


# Parameters
n_episodes = 5000
problem = 'CarRacing-v0'

gym.logger.set_level(40)
preview = False
best_result = 0
all_episode_reward = []

# Initialize simulation
env = gym.make(problem)
env.reset()
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release

solution = BaseSolution(env.action_space, model_outputs=2)


# Loop of episodes
for ie in range(n_episodes):
    state = env.reset()
    solution.reset()
    done = False
    episode_reward = 0
    no_reward_counter = 0

    # One-step-loop
    while not done:
        if preview:
            env.render()

        action, train_action = solution.get_action(state)
        new_state, reward, done, info = env.step(action)
        # TODO: Check if needed
        #reward = -100 if no_reward_counter > 200 else reward

        # Models action output has a different shape for this problem
        solution.learn(state, train_action, reward, new_state)
        state = new_state
        episode_reward += reward

        if reward < 0:
            no_reward_counter += 1
            if no_reward_counter > 200:
                break
        else:
            no_reward_counter = 0

    all_episode_reward.append(episode_reward)
    average_result = np.array(all_episode_reward[-10:]).mean()
    print('Average results:', average_result)

    if average_result > best_result:
        print('Saving best solution')
        solution.save_solution()
        best_result = average_result
