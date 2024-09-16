import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedPPOModel(nn.Module):
    def __init__(self, output_dim=6):
        super(EnhancedPPOModel, self).__init__()
        
        # Convolutional layers for processing the map inputs
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size after convolution to flatten the tensor
        conv_output_dim = 128 * 17 * 17
        player_info_dim = 4
        
        # Fully connected layers for combining map features with player information
        self.fc1 = nn.Linear(conv_output_dim + player_info_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, map_info, bomb_info):
        # Combine the two maps along the channel dimension
        # Pass the combined map through the convolutional layers
        x = F.relu(self.conv1(map_info))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output of the conv layers
        x = x.view(x.size(0), -1)
        
        # Concatenate the flattened map features with player information
        x = torch.cat((x, bomb_info), dim=1)
        
        # Pass the combined features through the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Return the output as a probability distribution over possible actions
        return F.softmax(x, dim=-1)
