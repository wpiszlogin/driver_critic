# driver_critic
Solution for CarRacing-v0 environment from OpenAI Gym. It uses the DDPG algorithm (Deep Deterministic Policy Gradient).

# Deep Deterministic Policy Gradient
The solution is composed from 4 Networks:
* Actor - play the game
* Critic - evalutate an Actor
* Target actor and Target Critic - produce target values for learning

![image](https://user-images.githubusercontent.com/6407844/111140756-ffdf5080-8582-11eb-8372-8764c0c0e1d9.png)

Reference:
https://arxiv.org/pdf/1509.02971.pdf

