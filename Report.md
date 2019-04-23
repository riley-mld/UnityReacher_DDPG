# Report
## Learning Algorithm
I used a DDPG (Deep Deterministic Policy Gradient) learning algorithm for this project. Two identical neural networks (identical in architecture) were used, one that does the learning and one set as target and soft update happens after a few time-steps of learning to make the target weights closer to the learning neural network. I also used double DQN, a dueling DQN and prioritized experience replay to improve the performance of the Agent. Although i ended up not using prioritized experience replay for the final result, as I wasn't able to get better results using prioritized experience replay.

### Hyper parameters
* Replay Buffer Size = 100000
* Batch Size = 256
* Gamma (Discount factor) = 0.99
* Tau (Soft update parameter) = 0.001
* Learning Rate for Actor Network = 0.001
* Learning Rate for Critic Network = 0.001
* Update Every = 20
* Number of Updates = 10
* number of episodes = 3000
* Epsilon Start = 1.0
* Epsilon Decay Rate =  0.000001
* Weight Decay = 0
* Noise Sigma = 0.05


### Model Architecture
Actor
`fc1 = [33, 256]`
`fc2 = [256, 256]`
`fc3 = [256, 4]`

Critic
`fc1 = [33, 256]`
`fc2 = [256, 256]`
`fc3 = [256, 1]`

## Plot of Rewards
Agent solved the environment in episode 36! And achieved a maximum average score over last 100 episode of 39 after training about 100 episodes.

![](Assets/reward_plot.png?raw=true)

## Ideas for Future Work
Implementing a Rainbow DQN and attempting to make the agent learn directly from the pixels.
