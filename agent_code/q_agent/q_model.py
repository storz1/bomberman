import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Environment parameters
GRID_SIZE = 17
ACTION_SPACE = 4  # Up, Down, Left, Right

# Q-Network: a simple feedforward network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Q-learning agent
class QLearningAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr

        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))  # Random action
        state = torch.FloatTensor(state.flatten()).unsqueeze(0)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state):
        state = torch.FloatTensor(state.flatten()).unsqueeze(0)
        next_state = torch.FloatTensor(next_state.flatten()).unsqueeze(0)
        target = reward

        target += self.gamma * torch.max(self.q_network(next_state)).item()

        current_q_value = self.q_network(state)[0][action]
        loss = self.criterion(current_q_value, torch.tensor(target))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_name):
        torch.save(self.q_network.state_dict(), file_name)
        print(f"Model saved to {file_name}")

    def load_model(self, file_name):
        self.q_network.load_state_dict(torch.load(file_name))
        self.q_network.eval()
        print(f"Model loaded from {file_name}")
        
    def init_randomly(self):
        for layer in self.q_network.children():
            nn.init.xavier_uniform_(layer.weight)