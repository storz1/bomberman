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
        # Check if the state or next_state is None or invalid
        if state is None or next_state is None:
            print("Warning: Invalid state or next_state, skipping this transition.")
            return
        self.memory.append((state, action, reward, next_state))

    def train(self):
        # Ensure there's enough data to train on
        if len(self.memory) == 0:
            print("No transitions to train on.")
            return

        for _ in range(self.k_epochs):
            try:
                states, actions, rewards, next_states = zip(*self.memory)

                # Convert memory to tensors, ensure compatibility
                states = torch.tensor(states, dtype=torch.float32, device=self.device)
                actions = torch.tensor(actions, dtype=torch.long, device=self.device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

                # Compute advantage estimates and returns
                returns = self.compute_returns(rewards)
                advantage = returns - self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

                # Filter out NaN values to prevent training issues
                advantage = advantage[~torch.isnan(advantage)]
                
                self.optimizer.zero_grad()
                action_probs = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                old_probs = action_probs.detach()
                ratio = action_probs / (old_probs + 1e-10)  # Prevent division by zero
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                loss = -torch.min(surr1, surr2).mean()
                loss.backward()
                self.optimizer.step()

            except Exception as e:
                print(f"Error during training loop: {e}")

        self.memory = []

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)
