import torch
import torch.optim as optim
from ppo import PPO
import events as e
import os
import time

def setup_training(self):
    """Initialize training-specific parameters."""
    
    K_epochs = 10
    eps_clip = 0.2
    gamma = 0.99
    
    lr_actor = 0.0003
    lr_critic = 0.001
    
    random_seed = 0

    self.log_dir = "PPO_logs"
    if not os.path.exists(self.log_dir):
        os.makedirs(self.log_dir)
    
    self.run_num = 0
    self.current_num_files = next(os.walk(self.log_dir))[2]
    self.run_num = len(self.current_num_files)
    self.log_f_name = self.log_dir + '/PPO_' + "log_" + str(self.run_num) + ".csv"

    print("current logging run number for : ", self.run_num)
    print("logging at : " + self.log_f_name)
    
    self.run_num_pretrained = 0
    self.directory = "PPO_preTrained"
    if not os.path.exists(self.directory):
        os.makedirs(self.directory)
    
    self.checkpoint_path = self.directory + "PPO_{}_{}.pth".format( random_seed, self.run_num_pretrained)
    
    self.model = PPO(player_info_dim = 4, action_dim = 6, lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip, device='cpu')
        

    
    # logging file
    self.log_f = open(self.log_f_name,"w+")
    self.log_f.write('episode,timestep,reward\n')



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
