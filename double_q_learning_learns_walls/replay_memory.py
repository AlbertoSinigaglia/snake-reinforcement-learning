import numpy as np


class ReplayMemoryFast:
    def __init__(self, memory_size, minibatch_size, subsample_size):
        self.subsample_size = subsample_size
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        self.experience = [None] * self.memory_size
        self.current_index = 0
        self.size = 0

    def store(self, observation, action, reward, newobservation):
        self.experience[self.current_index] = (np.copy(observation), np.copy(action), np.copy(reward), np.copy(newobservation))
        self.current_index += 1

        self.size = min(self.size + 1, self.memory_size)
        if self.current_index >= self.memory_size:
            self.current_index -= self.memory_size

    def sample(self):
        if self.size < self.minibatch_size:
            samples_index = np.arange(self.size)
        else:
            samples_index = np.random.choice(np.arange(self.size), self.minibatch_size, replace=False).astype(int)
        samples = []
        for i in samples_index:
            iter = self.experience[int(i)]
            if len(iter[0]) == self.subsample_size:
                indexes = np.arange(len(iter[0]))
            else:
                indexes = np.random.choice(np.arange(len(iter[0])), self.subsample_size, replace=False).astype(int)
            samples.append([el[indexes] for el in iter])
        return samples