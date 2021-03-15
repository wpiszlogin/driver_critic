"""
The solution was tested on a simple problem.
3 methods were overridden.
"""

import gym
import numpy as np

from tools import *
from base_solution import *

# TODO: Uncomment if there are no gym warnings
#gym.logger.set_level(40)


class TestSolution(BaseSolution):
    def preprocess(self, img, greyscale=True):
        return img

    def build_actor(self, state_shape, name="Actor"):
        inputs = layers.Input(shape=state_shape)
        x = inputs
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        last_init = tf.random_uniform_initializer(minval=-0.005, maxval=0.005)
        y = layers.Dense(self.model_action_out, activation='tanh', kernel_initializer=last_init)(x)
        y = y * 2

        model = Model(inputs=inputs, outputs=y, name=name)
        model.summary()
        return model

    def build_critic(self, state_shape, name="Critic"):
        state_inputs = layers.Input(shape=state_shape)
        x = state_inputs

        action_inputs = layers.Input(shape=(self.model_action_out,))
        x = layers.Dense(64, activation='relu')(x)
        x = layers.concatenate([x, action_inputs])

        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        y = layers.Dense(1)(x)

        model = Model(inputs=[state_inputs, action_inputs], outputs=y, name=name)
        model.summary()
        return model


# Parameters
n_episodes = 5000
problem = 'Pendulum-v0'

# Initialize simulation
env = gym.make(problem)
solution = TestSolution(env.action_space)

all_episode_reward = []

# Loop of episodes
for ie in range(n_episodes):
    # noinspection PyRedeclaration
    state = env.reset()
    solution.reset()
    done = False
    episode_reward = 0

    # One-step-loop
    while not done:
        #env.render()

        action, train_action = solution.get_action(state)
        new_state, reward, done, info = env.step(action)

        # Models action output has a different shape for this problem
        solution.learn(state, train_action, reward, new_state)
        state = new_state
        episode_reward += reward

    all_episode_reward.append(episode_reward)
    print('Average results:', np.array(all_episode_reward[-10:]).mean())
