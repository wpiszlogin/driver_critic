# driver critic
Solution for CarRacing-v0 environment from OpenAI Gym. It uses the DDPG algorithm (Deep Deterministic Policy Gradient).

## Quickstart
Dependencies:
* Gym 0.18.0
* Tensorflow 2.4.0
* Matplotlib 3.3.4

The current version of CarRacing-v0 has memory bugs. To solve it, we need to download the newest "car_racing.py" script from Gym GitHub.<br/>
Execute "main_loop.py" to train a new model.<br/>
It's possible to check the best solution by running "evaluate_loop.py".

# Solution
DDPG is composed of 4 Networks:
* Actor - play the game
* Critic - evaluate an Actor
* Target actor and Target Critic - produce target values for learning

![image](https://user-images.githubusercontent.com/6407844/111140756-ffdf5080-8582-11eb-8372-8764c0c0e1d9.png)

Reference:
https://arxiv.org/pdf/1509.02971.pdf

It was intended to make a base class that will be a foundation for every continuous-action task. It's easy to achieve more complex solutions, by inheriting base class.  CarRacing-v0 is a sort of computer vision problem, thus a convolution network was used.<br/>
The first implementation of R-buffer had functionality to avoid double-write of state and next state. To make sure it's not a problem it was simplified.

# Development
An investigation was made to solve a problem:
* The solution was adapted to Pendulum-v0 environment and learned successfully
* Agent could learn to accelerate or break if a reward was given for that
* Hyperparameter search: tau, gamma, learning rates, and parameters of noise generator. That was key 
* Different neural network architectures were tested
* Scale reward value

# Conclusion
DDPG is not an easy solution. The main goal was to tune a noise generator. It was noticed there are many successful DQN solutions. It's for discrete actions, but for some reason, it works better. One of the biggest challenge of the car was, that it loose control when turning and acceleration happens at the same time. In DQN case it's easy to avoid because we can define actions that exclude each other.

As future work, it is planned to:
* Use transfer learning: save to R buffer processed data from trained CNN network. Then build an RNN model from the map data.
* Check Proximal Policy Optimization.
