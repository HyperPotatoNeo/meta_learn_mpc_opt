# meta_learn_mpc_opt
This repo contains experiments to speed up differentiable MPC optimization by learning a meta-learner that creates a model which optimizes the cost function with very few iterations of backpropagation.

## Meta-Learners tested so far
* MAML
* Meta-LSTM

## Control tasks
* Double Integrator

## Credits
* https://github.com/markdtw/meta-learning-lstm-pytorch
* https://github.com/learnables/learn2learn
* https://github.com/facebookresearch/higher 