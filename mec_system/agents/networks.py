
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_alpha = nn.Linear(hidden_dim, action_dim)
        self.fc_beta = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Beta parameters must be > 0. Adding 1.0 ensures concentration around mode or uniform if near 1.
        alpha = F.softplus(self.fc_alpha(x)) + 1.0
        beta = F.softplus(self.fc_beta(x)) + 1.0
        return alpha, beta

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_val = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        val = self.fc_val(x)
        return val
