import torch
import torch.optim as optim
from .model import EnhancedPPOModel as PPOModel
from .ppo import PPOTrainer
import events as e

def setup_training(self):
    """Initialize training-specific parameters."""
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.logger.info('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = PPOModel().to(self.device)
    self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    self.trainer = PPOTrainer(self.model, self.optimizer, device=self.device)
    self.logger.info("PPO training initialized.")

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """Collect data for training."""
    reward = compute_reward(events)
    self.trainer.store_transition(old_game_state, self_action, reward, new_game_state)

def end_of_round(self, last_game_state, last_action, events):
    """Train the model after each round."""
    reward = compute_reward(events)
    self.trainer.store_transition(last_game_state, last_action, reward, None)
    self.trainer.train()

def compute_reward(events):
    """Compute reward based on the events that occurred."""
    reward = 0
    if e.COIN_COLLECTED in events:
        reward += 1
    if e.KILLED_OPPONENT in events:
        reward += 5
    if e.KILLED_SELF in events:
        reward -= 10
    return reward
