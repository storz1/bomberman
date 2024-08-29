import torch
import numpy as np
from .model import EnhancedPPOModel  # Import the correct model

def setup(self):
    """This is run once before the first game starts."""
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = EnhancedPPOModel(map_height=17, map_width=17, player_info_dim=12).to(self.device)
    self.model.eval()  # Set the model to evaluation mode
    self.logger.info("Enhanced PPO model loaded.")

def act(self, game_state: dict):
    """Called each step to determine the action."""
    map1, map2,  player_info = preprocess_game_state(game_state)
    
    # Convert the preprocessed state components to tensors and send to the correct device
    map1 = torch.tensor(map1, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(1)
    map2 = torch.tensor(map2, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(1)
    player_info = torch.tensor(player_info, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    with torch.no_grad():
        action_probs = self.model(map1, map2,  player_info)
    
    action = select_action(action_probs)
    return action

def preprocess_game_state(game_state):
    """Convert the game state to a format suitable for the model."""
    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    self_pos = game_state['self'][3]
    bomb_possible = game_state['self'][2]
    other_players = game_state['others']
    explosion_map = game_state['explosion_map']
    
    # Update the explosion map to include coins and bombs
    for i in range(len(coins)):
        if explosion_map[coins[i][0]][coins[i][1]] == 0:
            explosion_map[coins[i][0]][coins[i][1]] = -1
        
    for i in range(len(bombs)):
        if explosion_map[bombs[i][0][0]][bombs[i][0][1]] == 0:
            explosion_map[bombs[i][0][0]][bombs[i][0][1]] = bombs[i][1] + 10
    
    # Player positions and bomb possibilities
    positions = np.full(8, -10)
    positions[0] = self_pos[0]
    positions[1] = self_pos[1]
    
    bomb_possible_all = np.zeros(4)
    if bomb_possible:
        bomb_possible_all[0] = 1
    
    for i in range(len(other_players)):
        positions[2 + 2 * i] = other_players[i][3][0]
        positions[3 + 2 * i] = other_players[i][3][1]
        bomb_possible_all[i + 1] = other_players[i][2]
    
    # Combine positions and bomb_possible into a player_info vector
    player_info = np.concatenate([positions, bomb_possible_all])
    
    

    return field, explosion_map, player_info

def select_action(action_probs):
    """Select an action based on the action probabilities."""
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    action_index = torch.argmax(action_probs).item()
    return actions[action_index]
