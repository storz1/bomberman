import torch
import torch.optim as optim
from agent_code.actor_critic_easy.ppo import PPO_Easy
import events as e
import os
import time
import numpy as np

def setup_training(self):
    """Initialize training-specific parameters."""
    
    current_path_to_game = os.getcwd()
    
    K_epochs = 20
    eps_clip = 0.2
    gamma = 0.99
    
    lr_actor = 0.003#0.03
    lr_critic = 0.01

    self.log_dir = current_path_to_game +"/logs"
    if not os.path.exists(self.log_dir):
        os.makedirs(self.log_dir)
    
    self.run_num = 0
    self.current_num_files = next(os.walk(self.log_dir))[2]
    self.run_num = len(self.current_num_files)
    self.log_f_name = self.log_dir + '/PPO_' + "log_" + str(self.run_num) + ".csv"

    #print("current logging run number for : ", self.run_num)
    #print("logging at : " + self.log_f_name)
    
    self.run_num_pretrained = 0
    self.directory = current_path_to_game + "/preTrained"
    if not os.path.exists(self.directory):
        os.makedirs(self.directory)
        
    self.run_num_pretrained = next(os.walk(self.directory))[2]
    self.last_trained_model = len(self.run_num_pretrained)
    self.run_num_trainingsession = len(self.run_num_pretrained) +1
    
    self.checkpoint_path_load = self.directory + "/{}.pth".format( self.last_trained_model)
    self.checkpoint_path_save = self.directory + "/{}.pth".format( self.run_num_trainingsession)
    print("load model from path: " +str(self.checkpoint_path_load))
    print("save trained model to path : " +str(self.checkpoint_path_save))
    self.model = PPO_Easy(action_dim = 6, lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip, device='cpu')
    if(self.last_trained_model != 0):
        self.model.load(self.checkpoint_path_load)
    
    # logging file
    self.log_f = open(self.log_f_name,"w+")
    self.log_f.write('episode,timestep,reward\n')



def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """Collect data for training."""
    
    field = old_game_state['field']
    coins = old_game_state['coins']
    bombs = old_game_state['bombs']
    self_pos = old_game_state['self'][3]
    other_players = old_game_state['others']
    explosion_map = old_game_state['explosion_map']
    
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
        if(old_game_state['self'][2] == True):
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
        
    actions = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'])
    action_encoded = np.where(actions == self_action)[0]
    
    
    
    with torch.no_grad():
        state = torch.FloatTensor(features).unsqueeze(0).unsqueeze(1).to(self.device)
        action, action_logprob, state_val = self.model.policy_old.act(state)
            
    self.model.buffer.states.append(state)
    self.model.buffer.actions.append(torch.tensor(action_encoded))
    self.model.buffer.logprobs.append(action_logprob)
    self.model.buffer.state_values.append(state_val)
    
    
    
    reward = compute_reward_coin_pickup(events)
    self.model.buffer.rewards.append(reward)
    
    
    
    
    
def end_of_round(self, last_game_state, last_action, events):
    """Train the model after each round."""
    #Convert state into input of algorithm
    field = last_game_state['field']
    coins = last_game_state['coins']
    bombs = last_game_state['bombs']
    self_pos = last_game_state['self'][3]
    other_players = last_game_state['others']
    explosion_map = last_game_state['explosion_map']
    
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
        if(last_game_state['self'][2] == True):
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
        
    actions = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'])
    action_encoded = np.where(actions == last_action)[0]
    
    
    
    with torch.no_grad():
        state = torch.FloatTensor(features).unsqueeze(0).unsqueeze(1).to(self.device)
        action, action_logprob, state_val = self.model.policy_old.act(state)
            
    self.model.buffer.states.append(state)
    self.model.buffer.actions.append(torch.tensor(action_encoded))
    self.model.buffer.logprobs.append(action_logprob)
    self.model.buffer.state_values.append(state_val)
    
    
    
    reward = compute_reward_coin_pickup(events)
    self.model.buffer.rewards.append(reward)
    
    self.model.update()
    
    self.model.save(self.checkpoint_path_save)

def compute_reward_coin_pickup(events):
    """Compute reward based on the events that occurred."""
    reward = 0
    if e.COIN_COLLECTED in events:
        reward += 50
    if e.KILLED_SELF in events:
        reward -= 200
    #if e.BOMB_DROPPED in events:
    #    reward -= 10
    #if e.GOT_KILLED in events:
    #    reward -= 50
    if e.INVALID_ACTION in events:
        reward -= 100
    if e.WAITED in events:
        reward -= 10
        
    return reward
