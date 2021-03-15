# driver_critic
Solution for CarRacing-v0 environment from OpenAI Gym. It uses the DDPG algorithm (Deep Deterministic Policy Gradient).

## Quick start
Dependencies:
* Gym 0.18.0
* Tensorflow 2.4.0
* Matplotlib 3.3.4

Current version of CarRacing-v0 has memory bugs! To solve it, we need to downlowad newest "car_racing.py" script from Gym GitHub.
To run application execude "main_loop.py" script. 

# Solution
DDPG is composed from 4 Networks:
* Actor - play the game
* Critic - evalutate an Actor
* Target actor and Target Critic - produce target values for learning

![image](https://user-images.githubusercontent.com/6407844/111140756-ffdf5080-8582-11eb-8372-8764c0c0e1d9.png)

Reference:
https://arxiv.org/pdf/1509.02971.pdf

It was intended to make a base class that will be a foundation for every continous-action task. It's easy achive more complex solutions, by inherit base class.
CarRacing-v0 is a sort of computer vision problem, thus a convolution network was used. It was planned to extend the solution to transfer learning and RNN, but wasn't implemented because of deadline.


# Evaluation
Unfortunetly it was not possible to solve CarRacing-v0 problem.

An investigation was made to find a problem:
