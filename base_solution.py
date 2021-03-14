import numpy as np
from docutils.nodes import topic

from tools import *

"""
BaseSolution is class for every vision -> action problem.
It's based on the Deep Deterministic Policy Gradient algorithm.
It was intended to make a base class that will be a foundation for more complex solutions.
"""


class BaseSolution:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        self.noise = NoiseGenerator(np.full(action_space.shape, 0, np.float32),
                                    np.full(action_space.shape, 0.2, np.float32))
        # Initialize buffer R
        self.r_buffer = MemoriesRecorder(memory_capacity=40000)
        # TODO: Initialize critic and actor networks
        # TODO: Initialize target networks

    def reset(self):
        self.noise.reset()

    def preprocess(self, img, greyscale=True):
        if greyscale:
            img = img.mean(axis=2)
            img = np.expand_dims(img, 2)

        # Normalize from -1. to 1.
        img = (img / img.max()) * 2 - 1
        return img

    def get_action(self, state):
        # TODO: Result from network
        # TODO: Add noise
        # Temporary solution
        a = np.array([0.0, 1.0, 0.0])
        return a

    def learn(self, state, action, reward, new_state):
        # Store transition in R
        prep_state     = self.preprocess(state)
        prep_new_state = self.preprocess(new_state)
        self.r_buffer.write(prep_state, action, reward, prep_new_state)

        # Sample mini-batch from R
        state_batch, action_batch, reward_batch, new_state_batch  = self.r_buffer.sample()

        # TODO: Calc y
        # TODO: Update critic
        # TODO: Update actor
        # TODO: Update target networks
        pass
