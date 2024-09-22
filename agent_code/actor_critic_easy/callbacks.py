import torch
from agent_code.actor_critic_easy.model import ActorCriticEasy  # Import the correct model
import numpy as np

def setup(self):
    """This is run once before the first game starts."""
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.actor_critic = ActorCriticEasy(6).to(self.device)
    self.actor_critic.eval()  # Set the model to evaluation mode
    self.logger.info("Enhanced PPO model loaded.")

def act(self, game_state: dict):
    """Called each step to determine the action."""
    features = preprocess_game_state(game_state)
       
    # Convert the preprocessed state components to tensors and send to the correct device
    feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        action, action_logprob, state_val = self.actor_critic.act(feature_tensor)
    actions =  ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    
    return actions[action.item()]

def preprocess_game_state(game_state):
    """Convert the game state to a format suitable for the model."""
    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    self_pos = game_state['self'][3]
    other_players = game_state['others']
    explosion_map = game_state['explosion_map']
    
    Steps = [[1,0],[0,1],[-1,0],[0,-1]]
    #List of Features to be filled:
    features = np.zeros(15)

    # Input all the information into the field map
    self_pos_x = self_pos[0]
    self_pos_y = self_pos[1]
    
    # First Feature: Nearest Coin Delta x and y, Default = 0
    try:
        if(len(coins) != 0):
            distances = []
            for i in range(len(coins)):
                distances.append(np.sqrt((self_pos_x-coins[i][0])**2+(self_pos_y-coins[i][1])**2))
            #calculate lowest distance
            lowest_index = np.argmin(distances)
            features[0] = coins[lowest_index][0]-self_pos_x
            features[1] = coins[lowest_index][1]-self_pos_y
    except:
        print("Error in Coin Feature Extraction")
    
    # Second Feature: Delta x and y of all Opponents
    try:
        for i in range(len(other_players)):
            features[2+2*i] = other_players[i][3][0] - self_pos_x
            features[3+2*i] = other_players[i][3][1] - self_pos_y
    except:
        print("Error in Player Position Feature Extraction")
    
    
    # Third Feature: Surrounding Fields: free:0, Wall:-1, Danger:Timer Bomb, Death:-2
    for i in range(len(bombs)):
        field[bombs[i][0][0]][bombs[i][0][1]] = bombs[i][1] + 10
    try:
        count = 0
        for dx, dy in Steps:
            new_pos_x = self_pos_x + dx
            new_pos_y = self_pos_y + dy
            value_field = field[new_pos_x][new_pos_y]
            if(value_field ==-1 or value_field==1 or value_field >= 10):
                features[8+count] = -1
                continue
            if(explosion_map[new_pos_x][new_pos_y] == 1):
                features[8+count] = -2
                continue
            for i in range(len(bombs)):
                if(bombs[i][0][0] == new_pos_x):
                    distance = np.abs(new_pos_y - bombs[i][0][1])
                    explosion_timer = bombs[i][1]
                    if(distance <=4 and features[8+count] > explosion_timer):
                        features[8+count] = explosion_timer  
                elif(bombs[i][0][1] == new_pos_y):
                    distance = np.abs(new_pos_x - bombs[i][0][0])
                    explosion_timer = bombs[i][1]
                    if(distance <=4 and features[8+count] > explosion_timer):
                        features[8+count] = explosion_timer  
                
                        
                
            count +=1
    except Exception as err:
        print("Error in Surrounding Fields Feature Extraction")
        print(err)
    
    
    # Forth Feature: Value of dropping a bomb(amounts of crates destroyed and opponents in bomb area)
    
    try:
        for dx,dy in Steps:
            #two dimensions
            for length in range(1,4):
                #impact of bomb
                new_pos_x = self_pos_x + length * dx
                new_pos_y = self_pos_y + length * dy
                if(new_pos_x < 17 and new_pos_y < 17 and new_pos_x >= 0 and new_pos_y >= 0):
                    if(field[new_pos_x][new_pos_y] == 1):
                        features[12] += 1
                
                    for i in range(len(other_players)):
                        if(other_players[i][3][0]==new_pos_x and new_pos_y == other_players[i][3][1]):
                            features[12] += 4
                
    except:
        print("Error in Value of Bomb Feature Extraction")
    
    
    # Fifth Feature: Cooldown of own bomb
    try:
        if(game_state['self'][2] == True):
            features[13] = 1
    except:
        print("Error in Cooldown Feature of own bomb")
        
    # Sixth Feature: Dangerfaktor Standort
    try:
        for dx, dy in Steps:
            for length in range(1,4):
                new_pos_x = self_pos_x + length * dx
                new_pos_y = self_pos_y + length * dy
                if(new_pos_x < 17 and new_pos_y < 17 and new_pos_x >= 0 and new_pos_y >= 0):
                    for i in range(len(bombs)):
                        if(bombs[i][0][0] == new_pos_x and bombs[i][0][1] == new_pos_y):
                            features[14] += bombs[i][1]
                    
    except Exception as err:
        print(err)
        print("Error in Feature Extraction Danger in current place")
    
        
        
    return features

def select_action(action_probs):
    """Select an action based on the action probabilities."""
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    action_index = torch.argmax(action_probs).item()
    return actions[action_index]
