
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from mec_system.agents.networks import Actor, Critic
from mec_system.config import Config

class MAPPOAgent:
    def __init__(self, obs_dim, action_dim, device='cpu'):
        # Decentralized: each agent has its own actor and critic
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=Config.training.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=Config.training.LR_CRITIC)
        
        self.device = device
        self.clip_epsilon = Config.training.PPO_CLIP_EPSILON
        self.gamma = Config.training.GAMMA
        self.epochs = Config.training.PPO_EPOCHS
        self.entropy_coef = Config.training.ENTROPY_COEF
        
    def get_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print("NaN/Inf detected in observation!")
            print(f"Obs: {obs}")
            
        alpha, beta = self.actor(obs)
        
        if torch.isnan(alpha).any() or torch.isnan(beta).any():
            print("NaN detected in get_action actor output!")
            print(f"Alpha: {alpha}")
            print(f"Beta: {beta}")
            
        dist = torch.distributions.Beta(alpha, beta)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()
    
    def update(self, buffer):
        states, actions, rewards, next_states, old_log_probs, dones = zip(*buffer)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        
        # Compute Returns (Monte Carlo)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        total_actor_loss = 0
        total_critic_loss = 0
        
        for _ in range(self.epochs):
            # Compute current log probs and entropy
            if torch.isnan(states).any():
                print("NaN detected in states!")
            
            alpha, beta = self.actor(states)
            
            if torch.isnan(alpha).any() or torch.isnan(beta).any():
                print("NaN detected in actor output (alpha/beta)!")
                print(f"Alpha: {alpha}")
                print(f"Beta: {beta}")
                
            dist = torch.distributions.Beta(alpha, beta)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()  # Entropy for exploration
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute advantages (critic uses local state only)
            values = self.critic(states).squeeze()
            advantages = returns - values.detach()
            
            # PPO Clipped Objective with Entropy Bonus
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            # Critic Loss
            critic_loss = F.mse_loss(values, returns)
            
            # Update Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            
        return total_actor_loss / self.epochs, total_critic_loss / self.epochs
