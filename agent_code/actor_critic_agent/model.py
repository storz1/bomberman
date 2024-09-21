import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super(ActorCritic, self).__init__()


        # Actor network
        self.actor_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()#,
            #nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            #nn.ReLU()
        )
        self.actor_fc = nn.Sequential(
            nn.Linear(64*17 * 17, 64),  # Adjust input size as needed
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        
        # Critic network
        self.critic_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()#,
            #nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            #nn.ReLU()
        )
        self.critic_fc = nn.Sequential(
            nn.Linear(64* 17* 17, 64),  # Adjust input size as needed
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        

    def forward(self):
        raise NotImplementedError
    

    def act(self, state):#
        x = self.actor_conv(state)
        x = x.view(x.size(0), -1)  
        action_probs = self.actor_fc(x)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        # Critic forward pass
        x_critic = self.critic_conv(state)
        x_critic = x_critic.view(x_critic.size(0), -1)  # Flatten
        state_val = self.critic_fc(x_critic)
        
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        x = self.actor_conv(state)
        x = x.view(x.size(0), -1)  # Flatten
        action_probs = self.actor_fc(x)
        
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        x_critic = self.critic_conv(state)
        x_critic = x_critic.view(x_critic.size(0), -1)  # Flattenan
        state_values = self.critic_fc(x_critic)
        
        return action_logprobs, state_values, dist_entropy