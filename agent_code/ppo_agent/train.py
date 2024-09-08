import torch
import torch.optim as optim
from .model import EnhancedPPOModel
from .ppo import PPOTrainer
import events as e

def setup_training(self):
    """Initialize training-specific parameters."""
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = EnhancedPPOModel().to(self.device)
    self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    self.trainer = PPOTrainer(self.model, self.optimizer, device=self.device)
    self.logger.info("PPO training initialized.")

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """Collect data for training."""
    reward = compute_reward_coin_pickup(events)
    #print("old game state = " +str(old_game_state))
    #print("new game state = " +str(new_game_state))
    #print("action = " +str(self_action))
    #print("reward = " +str(reward))
    self.trainer.store_transition(old_game_state, self_action, reward, new_game_state)

def end_of_round(self, last_game_state, last_action, events):
    """Train the model after each round."""
    #reward = compute_reward_coin_pickup(events)
    #self.trainer.store_transition(last_game_state, last_action, reward, None)
    self.trainer.train()

def compute_reward_coin_pickup(events):
    """Compute reward based on the events that occurred."""
    reward = 0
    if e.COIN_COLLECTED in events:
        reward += 20
    if e.KILLED_SELF in events:
        reward -= 100
    if e.INVALID_ACTION in events:
        reward -= 10 
    if e.WAITED in events:
        reward -= 1
    
    
    return reward
