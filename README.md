# driver critic - Deep Deterministic Policy Gradient solution
Solution for CarRacing-v0 environment from OpenAI Gym. It uses Convolutional Neural Network for image processing and the Reinforcement Learning algorithm.
<br/><br/>

## Quickstart
Dependencies:
* Gym 0.18.0
* Tensorflow 2.4.0
* Matplotlib 3.3.4
<!---just
!The current version of CarRacing-v0 has memory bugs. To solve it, we need to download manually the newest "car_racing.py" script from Gym GitHub.<br/>
!https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
--->

Running application:
* Execute "main_loop.py" to train a new model. Press the SPACE key to watch a progress <br/>
* It's possible to review the best solution by running "evaluate_loop.py".

# Solution
The environment was challenging for a player as the car is very fast and not stable. The choice of Reinforcement Learning algorithms was limited according to the continuous characteristics of action data. Theoretically, Q-leaning could be implemented by assuming constant output values but it would reduce the precision of steering. Thus DDPG (Deep Deterministic Policy Gradient solution) was chosen as it is mix of Deep Q-Network and Deterministic Policy Gradient.

DDPG is composed of 4 Neural Networks:
* Actor - play the game
* Critic - evaluate an Actor
* Target actor and Target Critic - produce target values for learning

![actor_critic_graph_wide](https://user-images.githubusercontent.com/6407844/202601232-1fbbfa26-ce7c-43d7-8715-0925d0bde808.png)

![image](https://user-images.githubusercontent.com/6407844/111140756-ffdf5080-8582-11eb-8372-8764c0c0e1d9.png)

Reference:
https://arxiv.org/pdf/1509.02971.pdf

It was intended to make a base application that will be a foundation for every continuous-action task. It could be extended for more complex steering problems related to image processing. The CarRacing-v0 environment is a sort of computer vision problem, thus a convolution network was used.<br/>
A key component of the solution is a noise generator. This simple algorithm is responsible for exploring an environment. If generated actions don't make sense, then it will be hard to learn an agent. For example, it was important to avoid breaking and accelerating at the same time. For this reason, a network has only 2 outputs. Breaking and accelerating are in one axis. Thus using them simultaneously is prevented. A 'tanh' function was chosen as output activation, so by default model returns no actions or just a little.<br/>
Full training took 6h, but acceptable results can be achieved after 15 - 30 min (100 - 200 episodes).

# Development
An investigation was made to solve a problem:
* The solution was adapted to Pendulum-v0 environment and learned successfully
* Agent could learn to accelerate or break if a reward was given for that
* Hyperparameter search including noise generator
* Different neural network architectures were tested
* To speed up training an episode was interrupted if an agent didn't get a reward after 100 iterations

One of the biggest challenges of the car was, that it loose control when turning and acceleration happens at the same time. To avoid it the noise generator was upgraded so it avoids mixing of adverse actions.

# Preprocessing
* Hide numbers
* Enlarge speed information: bigger speed bar
* Road and grass has a uniform color
* Scale values from 0 to 1<br/><br/>
![image](https://user-images.githubusercontent.com/6407844/111695445-6a1e1c80-8833-11eb-869f-1c680784b658.png)

# Evaluation

The final solution is able to get an 848 average score. Generally, the results vary from 800 to 900 regarding track type. The agent keeps a vehicle in the track area. It has tended to stay near the right side. It's because it helps to go through the optimal racing line for left corners, which are most common on an Anti-clockwise circuit. Sometimes the controller is off-track on hairpin corners. Then it tries to go back, but not always succeed.
There is a room for development. Hyperparameters can be tuned more carefully. Also, we can implement an RNN network that should take advantage of time series data. <br/> <br/>
[![Watch the video](https://user-images.githubusercontent.com/6407844/111694067-aea8b880-8831-11eb-90b5-0d5396a6cba7.png)](https://youtu.be/_Olpk0Dt4gM)
<br/>

# Conclusion
DDPG is not an easy solution but can produce acceptable results if it's configured properly. The main goal was to tune a noise generator and simplify a task by limit action.

As future work, it is planned to:
* Implement auto-tuning of hyperparameters by Bayesian-optimization
* Use transfer learning: save to R buffer processed data from pretrained CNN network. Then build an RNN model from the map data.
* Test Proximal Policy Optimization.
