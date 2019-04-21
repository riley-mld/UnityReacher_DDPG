import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor Policy Network."""
    
    def __init__(self, state_size, action_size, seed, fc1_units=96, fc2_units=96):
        """Initialize the parameters and build the network.
        
        Params:
        state_size: Dimension of state
        action_size: Dimension of action
        seed: Random seed
        fc1_units: Number of nodes in the first hidden layer
        fc2_units: Number of nodes in the second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Model architecture
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        # Reset the weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset the weights"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e3, 3e-3)
        
    def forward(self, state):
        """Forward pass, this function outputs the action chosen by model."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
    
    
class Critic(nn.Module):
    """Critic Value Network."""
    
    def __init__(self, state_size, action_size, seed, fc1_units=96, fc2_units=96):
        """Initialize the parameters and build the network.
        
        Params:
        state_size: Dimension of state
        action_size: Dimension of action
        seed: Random seed
        fc1_units: Number of nodes in the first hidden layer
        fc2_units: Number of nodes in the second hidden layer        
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Model architecture
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        # Reset the weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset the weights"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e3, 3e3)
        
    def forward(self, state, action):
        """Forward pass, this function outputs the Q value of chosen action."""
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)