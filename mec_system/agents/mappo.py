
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from mec_system.agents.networks import Actor, Critic
from mec_system.config import *

class MAPPOAgent:
    def __init__(self, obs_dim, action_dim, device='cpu'):
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE_CRITIC)
        
        self.device = device
        self.clip_epsilon = PPO_CLIP_EPSILON
        self.gamma = GAMMA
        self.epochs = PPO_EPOCHS
        
    def get_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        mean, std = self.actor(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clip action to [0, 1] for environment, but keep raw for log_prob?
        # Usually we sample from Normal and then apply sigmoid, or just clip.
        # Since we used sigmoid on mean, the distribution is around [0,1].
        # Let's just clip the output for the env.
        
        return action.detach().cpu().numpy(), action_log_prob.detach().cpu().numpy()
    
    def update(self, buffer):
        # buffer: list of (obs, action, reward, next_obs, log_prob, done)
        # Unpack buffer
        obs = torch.FloatTensor(np.array([b[0] for b in buffer])).to(self.device)
        actions = torch.FloatTensor(np.array([b[1] for b in buffer])).to(self.device)
        rewards = torch.FloatTensor(np.array([b[2] for b in buffer])).to(self.device).unsqueeze(1)
        next_obs = torch.FloatTensor(np.array([b[3] for b in buffer])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array([b[4] for b in buffer])).to(self.device)
        dones = torch.FloatTensor(np.array([b[5] for b in buffer])).to(self.device).unsqueeze(1)
        
        # Compute Targets (Monte Carlo or TD)
        # Using simple TD target for now: r + gamma * V(s')
        with torch.no_grad():
            target_values = rewards + self.gamma * self.critic(next_obs) * (1 - dones)
            advantages = target_values - self.critic(obs)
            
            # Normalize advantages for stability (Convergence Assessment Recommendation)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # PPO Update
        for _ in range(self.epochs):
            # Recalculate log probs and values
            mean, std = self.actor(obs)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            
            curr_values = self.critic(obs)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages.squeeze()
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.squeeze()
            
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy # Add entropy bonus
            critic_loss = F.mse_loss(curr_values, target_values)
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Gradient clipping for stability (Convergence Assessment Recommendation)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()
            
        return actor_loss.item(), critic_loss.item()
