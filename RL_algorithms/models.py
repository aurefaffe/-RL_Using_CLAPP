import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, GELU, LeakyReLU, Softmax, Tanh, Identity
import random
from collections import deque

class ActorModel(nn.Module):
    def __init__(self, num_features, num_actions,*args, **kwargs):
        super().__init__(*args, **kwargs)
        

        self.layer = Linear(num_features, num_actions)
        self.softmax = Softmax(dim=-1)
        
    def forward(self, x, temp = None):
        if temp : 
            x = self.layer(x)/temp
        else:
           x = self.layer(x)
        x = self.softmax(x)
        return x
    
    

class CriticModel(nn.Module):

    def __init__(self, num_features, activation = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer = Linear(num_features, 1)
        
    
        if activation == 'ReLu':
            self.activation = ReLU()
        if activation == 'GELU':
            self.activation = GELU()
        if activation == "LeakyReLU":
            self.activation = LeakyReLU()
        else:
            print('activation not found: continuing without')
            self.activation = Identity()

        
    def forward(self, x):
       return self.activation(self.layer(x))








class Comparator(nn.Module):
    def __init__(self, num_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comparator = nn.Linear(num_features*2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, tocompare):
       combined = torch.cat((x, tocompare), dim=-1)
       return torch.sigmoid(self.comparator(combined))
    
class MemoryBuffer:
    def __init__(self, capacity = 10000, num_features=1024):
        self.capacity = capacity
        self.num_features = num_features
        self.buffer = deque(maxlen=capacity)
        self.features = deque(maxlen=capacity)
        self.visits = {}

    def add(self, obs, feature, similarity_threshold=0.9):
        obs_key = self.getobservation_key(obs)
        isnovel= True
        
        current_feat = feature.unsqueeze(0) if len(feature.shape) == 1 else feature
        similarities = []
        
        for stored_feat in self.features:
            stored_feat = stored_feat.unsqueeze(0) if len(stored_feat.shape) == 1 else stored_feat
            # Simple cosine similarity
            similarity = F.cosine_similarity(current_feat, stored_feat, dim=-1)
            similarities.append(similarity.item())
        
        # If too similar to existing observations, don't add
        if len(similarities) != 0 and max(similarities) > similarity_threshold:
            isnovel = False
        if isnovel:
            self.buffer.append(obs)
            self.features.append(feature.detach())
            self.visits[obs_key] = self.visits.get(obs_key, 0) + 1
        return isnovel
    
    def getobservation_key(self, obs):
        return hash(obs.detach().cpu().numpy().tobytes())
    
    def getvisits(self, obs):
        obs_key = self.getobservation_key(obs)
        return self.visits.get(obs_key, 0)
    
    def sample_batch(self, batch_size=32):
        """Sample batch of observations for training"""
        if len(self.buffer) < batch_size:
            return None
        
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch_obs = [self.buffer[i] for i in indices]
        batch_features = [self.features[i] for i in indices]
        
        return batch_obs, batch_features
    
class IntrinsicMotivationSystem():

    """Complete intrinsic motivation system"""
    def __init__(self, feature_dim=1024, memory_size=10000, 
                 alpha=0.5, beta=0.5, encoder=None):
        self.feature_dim = feature_dim
        self.alpha = alpha  # Count-based reward coefficient
        self.beta = beta    # Temporal distance reward coefficient
        
        # Networks
        self.encoder =encoder
        
        
        # Memory buffer
        self.memory = MemoryBuffer(memory_size, feature_dim)
        
        # Training data for temporal comparator
        self.training_buffer = deque(maxlen=50000)
 
        

    
    def compute_intrinsic_reward(self,obs, feats):
        """Compute intrinsic reward based on count-based and temporal distance"""

        
        # Count-based reward
        count_reward = self._compute_count_based_reward(feats)
        
        # # Temporal distance reward
        # temporal_reward = self._compute_temporal_distance_reward(feats)
        # print(f"Temporal distance reward: {temporal_reward}")
        
        # Combined intrinsic reward
        intrinsic_reward = self.alpha * count_reward # self.beta * temporal_reward
        
        # Add to memory if reward is high enough (novelty threshold)
        novelty_threshold = 0.24
        if intrinsic_reward > novelty_threshold:
            self.memory.add(obs, feats)
        
        # Store for temporal comparator training
        # self._store_training_data(obs, feats, episode_step)
        
        return intrinsic_reward
    
    def _compute_count_based_reward(self, features):
        """Compute count-based intrinsic reward"""
        
        
        # Find similar observations in memory
        visit_count = 0
        similarity_threshold = 0.8
        
        for stored_feat in self.memory.features:
            stored_feat = stored_feat.unsqueeze(0) if len(stored_feat.shape) == 1 else stored_feat
            similarity = F.cosine_similarity(features, stored_feat, dim=-1)
            
            if similarity.item() > similarity_threshold:
                visit_count += 10
        
        # Count-based reward: higher for less visited states
        count_reward = self.alpha / (visit_count + 1) 
        return count_reward
   
 

class SpatialRepresentation(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, place_cell_dim=256, hd_cell_dim=8, device = "cuda"):
        super().__init__()
        # Path integration component (simplified for bio-plausibility)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.place_cell_dim = place_cell_dim
        self.hd_cell_dim=hd_cell_dim
        self.device=torch.device(device)
        self.memory_update = nn.Linear(input_dim + hidden_dim, hidden_dim).to(device=device)
        
        # Place cell representation
        self.place_cells = nn.Linear(hidden_dim, place_cell_dim).to(device=device)
        
        # Head direction representation
        self.hd_cells = nn.Linear(hidden_dim, hd_cell_dim).to(device=device)
        
        self.hidden_state=None       
       
    
    def reset(self, batch_size=1):
        self.hidden_state = torch.zeros(batch_size, self.memory_update.out_features)
        
    def forward(self, visual_features):
        # Bio-plausible memory update (no backprop through time)
        with torch.no_grad():
            if self.hidden_state is None:
                self.reset(visual_features.size(0))       
            visual_features.to(self.device)
            new_state = (self.memory_update(
                torch.cat([visual_features.detach(), self.hidden_state.to(self.device)], dim=-1)
            ))
            
            # Generate spatial codes
            place_code = self.place_cells(new_state)
            hd_code = self.hd_cells(new_state)
            
            # Update state (with detached gradient to simulate biological constraints)
            self.hidden_state = new_state.detach().to(device=self.device)
            
        return torch.cat([place_code, hd_code], dim=-1)
    
class Predictor_Model(nn.Module):

    def __init__(self, action_dim, encoded_features_dim,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer = Linear(action_dim + encoded_features_dim, encoded_features_dim)

    def forward(self,encoded_features, action):
        return self.layer(torch.cat((encoded_features,action), dim= -1))
    







        