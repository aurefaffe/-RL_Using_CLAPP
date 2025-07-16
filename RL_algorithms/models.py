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
    
class IntrinsicMotivationSystem:
    """Complete intrinsic motivation system"""
    def __init__(self, feature_dim=1024, memory_size=10000, 
                 alpha=0.5, beta=0.5, encoder=None, device = 'cuda' ):
        self.feature_dim = feature_dim
        self.alpha = alpha  # Count-based reward coefficient
        self.beta = beta    # Temporal distance reward coefficient
        
        # Networks
        self.encoder =encoder
        self.comparator = Comparator(feature_dim).to(device=device)
        
        # Memory buffer
        self.memory = MemoryBuffer(memory_size, feature_dim)
        
        # Training data for temporal comparator
        self.training_buffer = deque(maxlen=50000)
        
        self.comparator_optimizer = torch.optim.Adam(self.comparator.parameters(), lr=1e-4)
        
        # Training parameters
        self.temporal_threshold = 5  # k steps for positive examples
        self.temporal_gap = 20       # M*k steps for negative examples
        

    
    def compute_intrinsic_reward(self,obs, feats, episode_step):
        """Compute intrinsic reward based on count-based and temporal distance"""

        
        # Count-based reward
        count_reward = self._compute_count_based_reward(feats)
        print(f"Count-based reward: {count_reward}")
        
        # Temporal distance reward
        temporal_reward = self._compute_temporal_distance_reward(feats)
        print(f"Temporal distance reward: {temporal_reward}")
        
        # Combined intrinsic reward
        intrinsic_reward = self.alpha * count_reward + self.beta * temporal_reward
        
        # Add to memory if reward is high enough (novelty threshold)
        novelty_threshold = 0.24
        if intrinsic_reward > novelty_threshold:
            self.memory.add(obs, feats)
        
        # Store for temporal comparator training
        self._store_training_data(obs, feats, episode_step)
        
        return intrinsic_reward
    
    def _compute_count_based_reward(self, features):
        """Compute count-based intrinsic reward"""
        
        
        # Find similar observations in memory
        visit_count = 0
        similarity_threshold = 0.8
        
        for stored_feat in self.memory.features:
            stored_feat = stored_feat.unsqueeze(0) if len(stored_feat.shape) == 1 else stored_feat
            similarity = F.cosine_similarity(features, stored_feat, dim=-1)
            print(f"Similarity: {similarity.item()}")
            
            if similarity.item() > similarity_threshold:
                visit_count += 10
        
        # Count-based reward: higher for less visited states
        count_reward = self.alpha / (visit_count + 1) 
        return count_reward
    
    def _compute_temporal_distance_reward(self, features):
        """Compute temporal distance based reward"""
        
        
        temporal_distance = 0.0
        
        for stored_feat in self.memory.features:
            print(features, stored_feat)
            # Use comparator to get temporal correlation
            temporal_correlation = self.comparator(features, stored_feat)
            
            # Convert correlation to distance (1 - correlation)
            temporal_distance = 1.0 - temporal_correlation
        
        # Reward proportional to minimum temporal distance
        temporal_reward = self.beta * temporal_distance
        return temporal_reward
    
    def _store_training_data(self, observation, features, episode_step):
        """Store data for training temporal comparator"""
        self.training_buffer.append((observation, features, episode_step))
    
    def train_temporal_comparator(self, batch_size=64):
        """Train the temporal comparator network"""
        if len(self.training_buffer) < batch_size*2:  # Changed > to <
            return

        # Sample training data
        batch_data = random.sample(self.training_buffer, batch_size*2)  # Added batch_size parameter

        positive_pairs = []
        negative_pairs = []

        for i in range(len(batch_data)):
            for j in range(i + 1, len(batch_data)):
                obs1, feat1, step1 = batch_data[i]
                obs2, feat2, step2 = batch_data[j]
                
                step_diff = abs(step1 - step2)
                
                if step_diff <= self.temporal_threshold:
                    # Positive pair (temporally close)
                    positive_pairs.append((feat1, feat2, 1.0))
                elif step_diff >= self.temporal_gap:
                    # Negative pair (temporally distant)
                    negative_pairs.append((feat1, feat2, 0.0))

        # Balance positive and negative pairs
        min_pairs = min(len(positive_pairs), len(negative_pairs))
        if min_pairs == 0:
            return

        positive_pairs = random.sample(positive_pairs, min_pairs)
        negative_pairs = random.sample(negative_pairs, min_pairs)

        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)

        # Training loop
        total_loss = 0
        self.comparator_optimizer.zero_grad()

        for feat1, feat2, label in all_pairs:
            label = torch.tensor([label], device=self.device, dtype=torch.float32)  # Added dtype
            
            prediction = self.comparator(feat1, feat2)
            prediction = prediction.squeeze(0)
            print(f"Prediction: {prediction}, Label: {label}")
            loss = F.binary_cross_entropy(prediction, label)
            loss.backward()
            total_loss += loss.item()

        self.comparator_optimizer.step()

        return total_loss / len(all_pairs)



    







        