import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import DQN


class Agent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.model = DQN(state_size, action_size)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01


    def choose_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state = torch.tensor(state, dtype=torch.float32)

        q_values = self.model(state)

        return torch.argmax(q_values).item()


    def train_step(self, state, action, reward, next_state, done):

        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        target = q_values.clone().detach()

        if done:
            target[action] = reward
        else:
            target[action] = reward + self.gamma * torch.max(next_q_values)

        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay