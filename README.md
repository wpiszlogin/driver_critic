# driver critic
Solution for CarRacing-v0 environment from OpenAI Gym. It uses the DDPG algorithm (Deep Deterministic Policy Gradient).

## Quickstart
Dependencies:
* Gym 0.18.0
* Tensorflow 2.4.0
* Matplotlib 3.3.4

The current version of CarRacing-v0 has memory bugs! To solve it, we need to download the newest "car_racing.py" script from Gym GitHub.
To run the application execute "main_loop.py" script.

# Solution
DDPG is composed of 4 Networks:
* Actor - play the game
* Critic - evaluate an Actor
* Target actor and Target Critic - produce target values for learning

![image](https://user-images.githubusercontent.com/6407844/111140756-ffdf5080-8582-11eb-8372-8764c0c0e1d9.png)

Reference:
https://arxiv.org/pdf/1509.02971.pdf

It was intended to make a base class that will be a foundation for every continuous-action task. It's easy to achieve more complex solutions, by inheriting base class.
CarRacing-v0 is a sort of computer vision problem, thus a convolution network was used. It was planned to extend the solution to transfer learning and RNN but wasn't implemented because of a deadline.
The first implementation of R-buffer had functionality to avoid double-write of state and next state. To make sure it's not a problem it was simplified.


# Evaluation
Unfortunately  it was not possible to solve CarRacing-v0 problem.
It's hard to see any progress in learning. It's strange because going forward should be easy to learn and by investigation we know the model was able to learn some specific actions. Maybe a way that reward is given doesn't fit DDPG.

An investigation was made to find a problem:
* The solution was adapted to Pendulum-v0 environment and learned successfully
* Agent could learn to accelerate or break if a reward was given for that
* Hyperparameter search: tau, gamma, learning rates, and parameters of noise generator
* Different neural network architectures were tested
* Scale reward value

# Conclusion
Probably DDPG was not the best choice for this problem. It is surprising because there are many DQN solutions, which can handle it even it uses discrete actions.
As future work, it is planned to check Proximal Policy Optimization.
