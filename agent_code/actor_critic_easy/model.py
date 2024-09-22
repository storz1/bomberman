import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticEasy(nn.Module):
    def __init__(self, action_dim):
        super(ActorCriticEasy, self).__init__()

        #Actor network
        self.actor = nn.Sequential(
            nn.Linear(15, 64),  # Adjust input size as needed
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(15, 64),  # Adjust input size as needed
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        

    def forward(self):
        raise NotImplementedError
    

    def act(self, state):
        action_probs = self.actor(state)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        # Critic forward pass
        state_val = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy