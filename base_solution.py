import numpy as np
from docutils.nodes import topic
import tensorflow as tf
from tools import *
from tensorflow.keras import layers
from tensorflow.keras import Model

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

        self.actor = None
        self.critic = None
        self.target_actor = None
        self.target_critic = None

    def reset(self):
        self.noise.reset()

    def build_actor(self, state_shape):
        inputs = layers.Input(shape=state_shape)
        x = inputs
        x = layers.Conv2D(32, kernel_size=(3, 3), padding='valid', use_bias=False, activation="relu")(inputs)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(64, kernel_size=(3, 3), padding='valid', use_bias=False, activation="relu")(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(64, kernel_size=(3, 3), padding='valid', use_bias=False, strides=(2, 2), activation="relu")(x)
        x = layers.AvgPool2D(pool_size=(2, 2))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        last_init = tf.random_uniform_initializer(minval=-0.005, maxval=0.005)
        y = layers.Dense(4, activation='sigmoid', kernel_initializer=last_init)(x)

        model = Model(inputs=inputs, outputs=y)
        print("Actor network:")
        model.summary()
        return model

    def init_networks(self, state_shape):
        self.actor = self.build_actor(state_shape)
        # TODO: Initialize critic
        # TODO: Initialize target networks

    def get_action(self, state):
        prep_state = self.preprocess(state)
        if self.actor is None:
            self.init_networks(prep_state.shape)

        # Get result from a network
        tensor_state = tf.expand_dims(tf.convert_to_tensor(prep_state), 0)
        actor_output = self.actor(tensor_state).numpy()

        # Decode output to actions
        actor_output = actor_output[0]
        action = np.array([actor_output[0] - actor_output[1], actor_output[2], actor_output[3]])

        # Add noise and clip min-max
        action += self.noise.generate()
        action = np.clip(np.array(action), a_min=self.action_space.low, a_max=self.action_space.high)
        return action

    def preprocess(self, img, greyscale=True):
        if greyscale:
            img = img.mean(axis=2)
            img = np.expand_dims(img, 2)

        # Normalize from -1. to 1.
        img = (img / img.max()) * 2 - 1
        return img

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

