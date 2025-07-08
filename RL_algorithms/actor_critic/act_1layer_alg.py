import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from load_standalone_model import load_model
# Option 1: CLAPP as Feature Extractor for Actor-Critic
class ActCrit1Layer(nn.Module):
    def __init__(self, env, clapp_model_path, gamma=0.99, freeze_clapp=True):
        super().__init__()
        self.env = env
        self.gamma = gamma
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLAPP model and move to device
        self.clapp_model = load_model(clapp_model_path, option=0).to(self.device)
        
        # Freeze CLAPP parameters if using as feature extractor
        if freeze_clapp:
            for param in self.clapp_model.parameters():
                param.requires_grad = False
        
        # Determine feature dimension from CLAPP
        with torch.no_grad():
            test_input = torch.randn(1, 1, 92, 92).to(self.device)  # Move test input to device
            test_features = self.clapp_model(test_input)
            feature_dim = test_features.shape[1]
        
        action_dim = env.action_space.n
        print(f"CLAPP feature dimension: {feature_dim}, Action dimension: {action_dim}")
        
        # Actor and Critic heads
        self.actor_fc = nn.Linear(feature_dim, action_dim).to(self.device)
        self.critic_fc = nn.Linear(feature_dim, 1).to(self.device)
        
        # For storing episode data
        self.rewards = []
        self.log_probs = []
        self.state_values = []
    
    
    def extract_features(self, state, keep_patches=False):
        state = torch.tensor(state, dtype=torch.float32)# Add batch dimension
          # Move state to the correct device
        state = state.view(state.shape[2], 1, state.shape[0], state.shape[1]).to(self.device) # Assuming state is (H, W, C)
        with torch.no_grad():
            features = self.clapp_model(state, all_layers=False, keep_patches=keep_patches)
        features = features[1] # Remove batch dimension if present
        return features.to(self.device)  # Ensure features are on the correct device
    
    def forward(self, state):
        features = self.extract_features(state)  # Ensure features are on the correct device
        return self.actor_fc(features), self.critic_fc(features)
    
    def act(self, state):
        """Select an action using CLAPP features"""
        # Extract features using CLAPP
        action = None
        
        
        # Get action probabilities and state value
        logits, state_value = self.forward(state)
        
        action_probs = F.softmax(logits, dim=-1)
        action_probs = action_probs.squeeze()  # Remove batch dimension if present
        dist = torch.distributions.Categorical(action_probs)
      # Remove batch dimension if present
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Store for training
        self.log_probs.append(log_prob)
        self.state_values.append(state_value.squeeze())
        
        return action.item()
    
    def clear_episode_data(self):
        self.rewards = []
        self.log_probs = []
        self.state_values = []
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def calculate_losses_and_update(self, optimizer):
        """Calculate losses and update model parameters"""
        if len(self.rewards) == 0:
            return 0.0
        
        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)  # Convert to tensor and move to device
        
        # Convert stored data to tensors
        log_probs = torch.stack(self.log_probs).to(self.device)  # Ensure log_probs are on the correct device
        state_values = torch.stack(self.state_values).to(self.device)  # Ensure state_values are on the correct device
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate advantages
        advantages = returns - state_values.detach()
        
        # Calculate losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        total_loss = actor_loss + critic_loss
        total_loss = total_loss.to(self.device) 
        
        # Update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
