import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOTrainer:
    def __init__(self, model, optimizer, device='cpu', gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.memory_field = []
        self.memory_bomb_possible = []
        self.memory_next_field = []
        self.memory_next_bomb_possible = []
        self.memory_actions = []
        self.memory_rewards = []
        

    def store_transition(self, state, action, reward, next_state):
        # Check if the state or next_state is None or invalid
        if state is None or next_state is None:
            print("Warning: Invalid state or next_state, skipping this transition.")
            return
        
        #Convert state into input of algorithm
        field = state['field']
        coins = state['coins']
        bombs = state['bombs']
        self_pos = state['self'][3]
        bomb_possible = state['self'][2]
        other_players = state['others']
        explosion_map = state['explosion_map']
        
        next_field = next_state['field']
        next_coins = next_state['coins']
        next_bombs = next_state['bombs']
        next_self_pos = next_state['self'][3]
        next_bomb_possible = next_state['self'][2]
        next_other_players = next_state['others']
        next_explosion_map = next_state['explosion_map']
        
        # Input all the information into the field map

        # First coins
        for i in range(len(coins)):
            field[coins[i][0]][coins[i][1]] = -5
            
        for i in range(len(next_coins)):
            next_field[next_coins[i][0]][next_coins[i][1]] = -5
            
        # Bombs
        for i in range(len(bombs)):
            field[bombs[i][0][0]][bombs[i][0][1]] = bombs[i][1] + 10
            
        for i in range(len(next_bombs)):
            next_field[next_bombs[i][0][0]][next_bombs[i][0][1]] = next_bombs[i][1] + 10
            
        #Explosions
        for i in range(len(explosion_map)):
            for j in range(len(explosion_map[i])):
                if(explosion_map[i][j]!=0):
                    field[i][j] = explosion_map[i][j]
                    
        for i in range(len(next_explosion_map)):
            for j in range(len(next_explosion_map[i])):
                if(next_explosion_map[i][j]!=0):
                    next_field[i][j] = next_explosion_map[i][j]
        
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
        
        try:
            next_field[next_self_pos[0]][next_self_pos[1]] = 20
        except:
            print("Own position error or player dead")
        
        try:
            for i in range(len(next_other_players)):
                next_field[next_other_players[i][3][0]][next_other_players[i][3][1]] = 21 + i
        except:
            print("Error with position from other players")
            
        # Second input: is bomb possible for players    
        bomb_possible_all = np.zeros(4)
        if bomb_possible:
            bomb_possible_all[0] = 1
        
        for i in range(len(other_players)):
            bomb_possible_all[i + 1] = other_players[i][2]
        
        next_bomb_possible_all = np.zeros(4)
        if next_bomb_possible:
            next_bomb_possible_all[0] = 1
        
        for i in range(len(next_other_players)):
            next_bomb_possible_all[i + 1] = next_other_players[i][2]
  
        
        
        actions = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'])
        action_encoded = np.where(actions == action)[0]+1
        
        
        self.memory_field.append(torch.tensor(np.array(field), dtype=torch.float32, device=self.device).unsqueeze(0))
        self.memory_bomb_possible.append(torch.tensor(np.array(bomb_possible_all), dtype=torch.float32, device=self.device))
        self.memory_next_field.append(torch.tensor(np.array(next_field), dtype=torch.float32, device=self.device).unsqueeze(0))
        self.memory_next_bomb_possible.append(torch.tensor(np.array(next_bomb_possible_all), dtype=torch.float32, device=self.device))
        self.memory_actions.append(action_encoded)
        self.memory_rewards.append(reward)
        
        #print(self.memory_bomb_possible[-1].shape)

    def train(self):
        # Ensure there's enough data to train on
        if len(self.memory_rewards) == 0:
            print("No transitions to train on.")
            return

        for _ in range(self.k_epochs):
            try:
                #first calculate return for all rewards
                torch_return = self.compute_returns(self.memory_rewards)
             
                #second transfer all data into torch tensors
                torch_field = torch.stack(self.memory_field)
                torch_bomb_possible = torch.stack(self.memory_bomb_possible)
                torch_actions = torch.tensor(np.array(self.memory_actions), dtype=torch.int64, device=self.device)
                advantage = (torch_return - self.model(torch_field, torch_bomb_possible).gather(1, torch_actions.squeeze(-1).unsqueeze(-1)))
                
            
                # Filter out NaN values to prevent training issues
                advantage = advantage[~torch.isnan(advantage)]
                
                self.optimizer.zero_grad()
                action_probs = self.model(torch_field, torch_bomb_possible).gather(1, torch_actions.squeeze(-1).unsqueeze(-1))
                old_probs = action_probs.detach()
                ratio = action_probs / (old_probs + 1e-10)  # Prevent division by zero
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                loss = -torch.min(surr1, surr2).mean()
                loss.backward()
                self.optimizer.step()
                                
                
                
            except Exception as e:
                print(f"Error during training loop: {e}")

        self.memory_field = []
        self.memory_bomb_possible = []
        self.memory_next_field = []
        self.memory_next_bomb_possible = []
        self.memory_actions = []
        self.memory_rewards = []

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)
