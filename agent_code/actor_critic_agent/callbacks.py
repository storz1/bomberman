import torch
import numpy as np
from .model import EnhancedPPOModel  # Import the correct model

def setup(self):
    """This is run once before the first game starts."""
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = EnhancedPPOModel().to(self.device)
    self.model.eval()  # Set the model to evaluation mode
    self.logger.info("Enhanced PPO model loaded.")

def act(self, game_state: dict):
    """Called each step to determine the action."""
    map_info ,  bomb_info = preprocess_game_state(game_state)
    
    # Convert the preprocessed state components to tensors and send to the correct device
    map_info = torch.tensor(map_info, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(1)
    bomb_info = torch.tensor(bomb_info, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    with torch.no_grad():
        action_probs = self.model(map_info,  bomb_info)
    
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

    # Input all the information into the field map

    # First coins
    for i in range(len(coins)):
        field[coins[i][0]][coins[i][1]] = -5
        
    # Bombs
    for i in range(len(bombs)):
        field[bombs[i][0][0]][bombs[i][0][1]] = bombs[i][1] + 10
        
    #Explosions
    for i in range(len(explosion_map)):
        for j in range(len(explosion_map[i])):
            if(explosion_map[i][j]!=0):
                field[i][j] = explosion_map[i][j]
    
    # Players
    try:
        field[self_pos[0]][self_pos[1]] = 20
    except:
        print("Own position error or player dead")
    
    try:
        for i in range(len(other_players)):
            field[other_players[i][3][0]][other_players[i][3][1]] = 21 + i
    except:
        print("Error with position from other players")
        
    # Second input: is bomb possible for players    
    bomb_possible_all = np.zeros(4)
    if bomb_possible:
        bomb_possible_all[0] = 1
    
    for i in range(len(other_players)):
        bomb_possible_all[i + 1] = other_players[i][2]
     
        
    return field, bomb_possible_all

def select_action(action_probs):
    """Select an action based on the action probabilities."""
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    action_index = torch.argmax(action_probs).item()
    return actions[action_index]
