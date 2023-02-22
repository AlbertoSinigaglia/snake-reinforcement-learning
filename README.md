# Snake
This repository contains an attempt to learn to play snake using reinforcement learning using function approximation via Deep Learning.  

The project is divided in:
 - fully observable MDP, using Deep Q-learning, Double Deep Q-learning, Duelling Deep Q-learning, and finally A2C
 - partially observable MDP, using A2C, made "PO"MDP allowing the agent to see only the neighborhood of the head. 


Everything is implemented from scratch using Tensorflow for the models and Numpy for the environments.

## Training
Full training using DDQN

https://user-images.githubusercontent.com/25763924/213567932-478bdaaa-6dfd-4b6a-b577-01decaae5235.mp4

## Evaluation
Evaluation of the model trained using DDQN

https://user-images.githubusercontent.com/25763924/213567968-ea245697-2e7a-495b-b302-c73dc53d91c8.mp4

## Comparison
Comparison of A2C on fully observable, and partially observable (5x5 window and 3x3 window)

https://user-images.githubusercontent.com/25763924/220478593-06081b09-35fc-4a2e-b8d1-a9031a4766fc.mp4

