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
        # The result is based on an old value
        # The second segment will keep values near a mean value
        # It uses normal distribution multiplied by a standard deviation
        self.x = self.x \
                 + self.theta * (self.mean - self.x) * self.dt \
                 + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.x_shape)

        return self.x


"""
Class for maintaining memories of a solution.
"""


class MemoryRecorder:
    def __init__(self):
        # TODO: init recorder
        pass

    def write(self):
        # TODO: write
        pass

    def sample(self):
        # TODO: sample
        pass


"""
Helps to visualize any type of image with a colorbar.
"""


def show_img(img):
    if len(img.shape) < 3 or img.shape[2] == 1:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.colorbar()
