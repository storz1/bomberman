import torch
import torch.optim as optim
from agent_code.actor_critic_agent.ppo import PPO
import events as e
import os
import time
import numpy as np

def setup_training(self):
    """Initialize training-specific parameters."""
    
    current_path_to_game = os.getcwd()
    
    K_epochs = 20
    eps_clip = 0.8
    gamma = 0.99
    
    lr_actor = 0.1#0.03
    lr_critic = 0.2

    self.log_dir = current_path_to_game +"/PPO_logs"
    if not os.path.exists(self.log_dir):
        os.makedirs(self.log_dir)
    
    self.run_num = 0
    self.current_num_files = next(os.walk(self.log_dir))[2]
    self.run_num = len(self.current_num_files)
    self.log_f_name = self.log_dir + '/PPO_' + "log_" + str(self.run_num) + ".csv"

    #print("current logging run number for : ", self.run_num)
    #print("logging at : " + self.log_f_name)
    
    self.run_num_pretrained = 0
    self.directory = current_path_to_game + "/PPO_preTrained"
    if not os.path.exists(self.directory):
        os.makedirs(self.directory)
        
    self.run_num_pretrained = next(os.walk(self.directory))[2]
    self.last_trained_model = len(self.run_num_pretrained)
    self.run_num_trainingsession = len(self.run_num_pretrained) +1
    
    self.checkpoint_path_load = self.directory + "/PPO_{}.pth".format( self.last_trained_model)
    self.checkpoint_path_save = self.directory + "/PPO_{}.pth".format( self.run_num_trainingsession)
    print("load model from path: " +str(self.checkpoint_path_load))
    print("save trained model to path : " +str(self.checkpoint_path_save))
    self.model = PPO(action_dim = 6, lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip, device='cpu')
    if(self.last_trained_model != 0):
        self.model.load(self.checkpoint_path_load)

    
    # logging file
    self.log_f = open(self.log_f_name,"w+")
    self.log_f.write('episode,timestep,reward\n')



def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """Collect data for training."""
    
    #data preprocessing to save data as tensor
    
    #Convert state into input of algorithm
    field = old_game_state['field']
    coins = old_game_state['coins']
    bombs = old_game_state['bombs']
    self_pos = old_game_state['self'][3]
    other_players = old_game_state['others']
    explosion_map = old_game_state['explosion_map']
    
    for i in range(len(coins)):
        field[coins[i][0]][coins[i][1]] = -5
        
    for i in range(len(bombs)):
        field[bombs[i][0][0]][bombs[i][0][1]] = bombs[i][1] + 10
        
    for i in range(len(explosion_map)):
        for j in range(len(explosion_map[i])):
            if(explosion_map[i][j]!=0):
                field[i][j] = explosion_map[i][j]
    
    try:
        field[self_pos[0]][self_pos[1]] = 20
    except:
        print("Own position error or player dead")
    
    try:
        for i in range(len(other_players)):
            field[other_players[i][3][0]][other_players[i][3][1]] = 21 + i
    except:
        print("Error with position from other players")
        
    actions = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'])
    action_encoded = np.where(actions == self_action)[0]
    
    
    
    with torch.no_grad():
        state = torch.FloatTensor(field).unsqueeze(0).unsqueeze(1).to(self.device)
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
    
    for i in range(len(coins)):
        field[coins[i][0]][coins[i][1]] = -5
        
    for i in range(len(bombs)):
        field[bombs[i][0][0]][bombs[i][0][1]] = bombs[i][1] + 10
        
    for i in range(len(explosion_map)):
        for j in range(len(explosion_map[i])):
            if(explosion_map[i][j]!=0):
                field[i][j] = explosion_map[i][j]
    
    try:
        field[self_pos[0]][self_pos[1]] = 20
    except:
        print("Own position error or player dead")
    
    try:
        for i in range(len(other_players)):
            field[other_players[i][3][0]][other_players[i][3][1]] = 21 + i
    except:
        print("Error with position from other players")
        
    actions = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'])
    action_encoded = np.where(actions == last_action)[0]
    
    
    
    with torch.no_grad():
        state = torch.FloatTensor(field).unsqueeze(0).unsqueeze(1).to(self.device)
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
    reward = 10
    if e.COIN_COLLECTED in events:
        reward += 50
    if e.KILLED_SELF in events:
        reward -= 200
    if e.BOMB_DROPPED in events:
        reward += 5
    #if e.GOT_KILLED in events:
    #    reward -= 50
    if e.INVALID_ACTION in events:
        reward -= 100
    if e.WAITED in events:
        reward -= 10
        
    return reward
