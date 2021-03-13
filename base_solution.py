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
        # TODO: initialize buffer R
        # TODO: initialize critic and actor networks
        # TODO: initialize target networks

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
        # TODO: result from network
        # TODO: add noise
        # Temporary solution
        a = np.array([0.0, 1.0, 0.0])
        return a

    def learn(self, state, action, reward, new_state):
        # TODO: store transition in R
        # TODO: sample mini-batch from R
        # TODO: calc y
        # TODO: update critic
        # TODO: update actor
        # TODO: update target networks
        pass
