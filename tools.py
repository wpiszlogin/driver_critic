from matplotlib import pyplot as plt
import numpy as np

"""
Noise generator that helps to explore action in DDPG.
Values are chosen by Ornstein–Uhlenbeck algorithm.
"""


class NoiseGenerator:
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2):
        self.theta = theta
        self.dt = dt
        self.mean = mean
        self.std_dev = std_dev

        if mean.shape != std_dev.shape:
            raise ValueError('Mean shape: {} and std_dev shape: {} should be the same!'.format(
                mean.shape, std_dev.shape))

        # This shape will be generated
        self.x_shape = mean.shape
        self.x = None

        self.reset()

    def reset(self):
        # Reinitialize generator
        self.x = np.zeros_like(self.x_shape)

    def generate(self):
        # The result is based on the old value
        # The second segment will keep values near a mean value
        # It uses normal distribution multiplied by a standard deviation
        self.x = (self.x
                  + self.theta * (self.mean - self.x) * self.dt
                  + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.x_shape))

        return self.x


"""
Class for maintaining memories of an agent.
It allows writing a single record to a database and sample from it.
"""


class MemoriesRecorder:
    def __init__(self, memory_capacity=50000):
        self.memory_capacity = memory_capacity

        # Memory will be initialized when first time used
        self.uni_state_db = None
        self.action_db    = None
        self.reward_db    = None

        self.writes_num = 0

    def init_memory(self, state_shape, action_shape):
        state_shape  = prepend_tuple(self.memory_capacity, state_shape)
        action_shape = prepend_tuple(self.memory_capacity, action_shape)

        self.uni_state_db = np.zeros(state_shape, np.float32)
        self.action_db    = np.zeros(action_shape, np.float32)
        self.reward_db    = np.zeros(self.memory_capacity, np.float32)

    def write(self, state, action, reward, new_state):
        if self.uni_state_db is None:
            self.init_memory(state.shape, action.shape)

        # Write indexes
        memory_index = self.writes_num     % self.memory_capacity
        next_index   = (self.writes_num + 1) % self.memory_capacity

        # Save next state to the same array with a next index
        self.uni_state_db[memory_index] = state
        self.action_db[memory_index]    = action
        self.reward_db[memory_index]    = reward
        self.uni_state_db[next_index]   = new_state

        self.writes_num += 1

    def sample(self, batch_size=64):
        all_indexes = np.arange(min(self.memory_capacity, self.writes_num))

        # Writing index points to a record that is always corrupted
        # It's better to replace it with a neighbor index
        current_index        = self.writes_num % self.memory_capacity
        next_current_index   = (self.writes_num + 1) % self.memory_capacity
        all_indexes = np.where(all_indexes == current_index, next_current_index, all_indexes)

        sampled_indexes = np.random.choice(all_indexes, batch_size)
        next_indexes = (sampled_indexes + 1) % self.memory_capacity

        return (self.uni_state_db[sampled_indexes],
                self.action_db[sampled_indexes],
                self.reward_db[sampled_indexes],
                self.uni_state_db[next_indexes])


"""
Helps to visualize any type of image with a colorbar.
"""


def show_img(img):
    if len(img.shape) < 3 or img.shape[2] == 1:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.colorbar()


"""
Inserts a new element to the beginning of a tuple.
"""


def prepend_tuple(new_dim, some_shape):
    some_shape_list = list(some_shape)
    some_shape_list.insert(0, new_dim)
    return tuple(some_shape_list)
