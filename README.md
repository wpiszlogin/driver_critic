# driver critic
Solution for CarRacing-v0 environment from OpenAI Gym. It uses the DDPG algorithm (Deep Deterministic Policy Gradient).

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
* Change colors to grayscale to limit computation
* Hide numbers
* Enlarge speed information: bigger speed bar
* Road and grass has a uniform color
* Car color is black, to make it more visible
![image](https://user-images.githubusercontent.com/6407844/111527315-1262b100-8760-11eb-8908-10ce5b13a8a0.png)

# Evaluation
The final solution is able to get an 802 average score. Results vary from 750 to 850 regarding track type. The controller makes sometimes errors on hairpin corners (that are most challenging). In this situation, it is able to go back on track.<br>
There is room for development. Hyperparameters can be tuned more carefully. Also, we can implement an RNN network that should take advantage of time series data.

# Conclusion
DDPG is not an easy solution but can produce acceptable results if it's configured properly. The main goal was to tune a noise generator and simplify a task by limit action.

As future work, it is planned to:
* Implement auto-tuning of hyperparameters by Bayesian-optimization
* Use transfer learning: save to R buffer processed data from pretrained CNN network. Then build an RNN model from the map data.
* Check Proximal Policy Optimization.
