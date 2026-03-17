import random
import numpy as np

class SnakeEnv:

    def __init__(self):
        self.state_size = 15
        self.max_steps = 100
        self.steps = 0

    def reset(self):
        self.steps = 0
        state = np.random.rand(self.state_size)
        return state

    def step(self, action):

        self.steps += 1

        next_state = np.random.rand(self.state_size)

        reward = random.choice([-1, 0, 1])

        done = self.steps >= self.max_steps

        return next_state, reward, done