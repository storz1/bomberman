from .q_model import QNetwork, QLearningAgent

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


def setup(self):
    np.random.seed()
    
    self.model = QLearningAgent(state_size=17 * 17, action_size=4)
    
    self.model.load_model("q_learning_agent.pth")
    #self.model.init_randomly()
    


# Function to obtain action from a trained model and a given game state
def get_action_from_state(agent, game_state):
    
    state = torch.FloatTensor(game_state.flatten()).unsqueeze(0)
    
    with torch.no_grad():
        q_values = agent.q_network(state)
        
    action = torch.argmax(q_values).item()
    return action
    

def act(agent, game_state: dict):
    #agent.logger.info()
    
    action_index = get_action_from_state(agent.model, game_state["field"])
    action = ["UP","DOWN","LEFT","RIGHT"][action_index]
   
    return action
