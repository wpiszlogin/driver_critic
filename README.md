# driver critic - Deep Deterministic Policy Gradient solution
Solution for CarRacing-v0 environment from OpenAI Gym. It uses the DDPG algorithm for reinforcement learning.
<br/><br/>
[![Watch the video](https://user-images.githubusercontent.com/6407844/111694067-aea8b880-8831-11eb-90b5-0d5396a6cba7.png)](https://youtu.be/_Olpk0Dt4gM)
<br/>
## Quickstart
Dependencies:
* Gym 0.18.0
* Tensorflow 2.4.0
* Matplotlib 3.3.4

The current version of CarRacing-v0 has memory bugs. To solve it, we need to download manually the newest "car_racing.py" script from Gym GitHub.<br/>
https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py

Running application:
* Execute "main_loop.py" to train a new model. Press the SPACE key to watch a progress <br/>
* It's possible to check the best solution by running "evaluate_loop.py".

# Solution
DDPG is composed of 4 Networks:
* Actor - play the game
* Critic - evaluate an Actor
* Target actor and Target Critic - produce target values for learning

![actor_critic_graph](https://user-images.githubusercontent.com/6407844/202599878-c09878e0-6cc1-4f75-929a-2d2c5f137e21.png =250x250)

![image](https://user-images.githubusercontent.com/6407844/111140756-ffdf5080-8582-11eb-8372-8764c0c0e1d9.png)

Reference:
https://arxiv.org/pdf/1509.02971.pdf

It was intended to make a base class that will be a foundation for every continuous-action task. It's easy to achieve more complex solutions, by inheriting base class.  CarRacing-v0 is a sort of computer vision problem, thus a convolution network was used.<br/>
A key component of the solution is a noise generator. This simple algorithm is responsible for exploring an environment. If generated actions don't make sense, then it will be hard to learn an agent. For example, it was important to avoid breaking and accelerating at the same time. For this reason, a network has only 2 outputs. Breaking and accelerating are in one axle. Thus using them simultaneously is prevented. A 'tanh' function was chosen as output activation, so by default model returns no actions or just a little.<br/>
One of the most important things was to simplify the task. The car is very fast and not stable. To make it more user friendly all action was divided by 4, so acceleration, braking, and turning were much limited.<br/>
Full training took 6h, but acceptable results can be achieved after 15 - 30 min (100 - 200 episodes).

# Development
An investigation was made to solve a problem:
* The solution was adapted to Pendulum-v0 environment and learned successfully
* Agent could learn to accelerate or break if a reward was given for that
* Hyperparameter search: especially about noise generator, a few changes in learning rates
* Different neural network architectures were tested
* Episode was interrupted if an agent didn't get a reward after 100 iterations

It was noticed there are many DQN solutions. It's for discrete actions, but for some reason, it works better. One of the biggest challenges of the car was, that it loose control when turning and acceleration happens at the same time. In DQN case it's easy to avoid because we can define actions that exclude each other.

# Preprocessing
* Hide numbers
* Enlarge speed information: bigger speed bar
* Road and grass has a uniform color
* Scale values from 0 to 1
![image](https://user-images.githubusercontent.com/6407844/111695445-6a1e1c80-8833-11eb-869f-1c680784b658.png)

# Evaluation

The final solution is able to get an 848 average score. Generally, the results vary from 800 to 900 regarding track type. The agent keeps a vehicle in the track area. It has tended to stay near the right side. It's because it helps to go through the optimal racing line for left corners, which are most common on an Anti-clockwise circuit. Sometimes the controller is off-track on hairpin corners. Then it tries to go back, but not always succeed.
There is room for development. Hyperparameters can be tuned more carefully. Also, we can implement an RNN network that should take advantage of time series data. <br/> <br/>

# Conclusion
DDPG is not an easy solution but can produce acceptable results if it's configured properly. The main goal was to tune a noise generator and simplify a task by limit action.

As future work, it is planned to:
* Implement auto-tuning of hyperparameters by Bayesian-optimization
* Use transfer learning: save to R buffer processed data from pretrained CNN network. Then build an RNN model from the map data.
* Check Proximal Policy Optimization.
