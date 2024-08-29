import torch
import torch.nn as nn
import torch.optim as optim

class PPOTrainer:
    def __init__(self, model, optimizer, device='cpu', gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.memory = []

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train(self):
        for _ in range(self.k_epochs):
            states, actions, rewards, next_states = zip(*self.memory)

            # Convert memory to tensors
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

            # Compute advantage estimates and returns
            returns = self.compute_returns(rewards)
            advantage = returns - self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

            # Compute loss and update model
            self.optimizer.zero_grad()
            action_probs = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            old_probs = action_probs.detach()
            ratio = action_probs / old_probs
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2).mean()
            loss.backward()
            self.optimizer.step()

        self.memory = []

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)
