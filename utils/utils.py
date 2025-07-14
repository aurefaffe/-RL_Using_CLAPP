import miniworld.wrappers
import torch
import math
import mlflow
import argparse
import miniworld
import os
import numpy as np 
import gymnasium as gym
import random
from collections import deque
import torch.nn.functional as F
import torch.nn as nn


from mlflow import MlflowClient, MlflowException




def parsing():
    parser = argparse.ArgumentParser()
    #arguments for the environment
    parser.add_argument('--environment',default='T_maze/custom_T_Maze_V0.py', help= 'name of the environment')
    parser.add_argument('--greyscale', action= 'store_true', help = 'determine if we keep render the state in greyscale')
    parser.add_argument('--render', action= 'store_true', help= 'will render the maze')
    parser.add_argument('--num_envs', type= int ,default= 8, help= 'the number of synchronous environment to spawn')

    #arguments for the training
    parser.add_argument('--algorithm',default= 'PPO', help= 'type of RL algorithm to use')
    parser.add_argument('--encoder', default= "CLAPP", help="decide which encoder to use")
    parser.add_argument('--seed', default= 0, type= int, help= 'manual seed for training')
    parser.add_argument('--checkpoint_interval', default= 50, type= int, help= 'interval at which to save the model weights')


    #hyperparameters for the training
    parser.add_argument('--num_epochs', default= 1800, help= 'number of epochs for the training')
    parser.add_argument('--len_rollout', default= 1024, help= 'length of the continuous rollout')
    parser.add_argument('--num_updates', default= 16, help= 'number of steps for the optimizer')
    parser.add_argument('--minibatch_size', default= 64, help= 'define minibatch size for offline learning')
    parser.add_argument('--actor_lr', default= 5e-3, help= 'learning rate for the actor if the algorithm is actor critic')
    parser.add_argument('--critic_lr', default= 1e-3, help= 'learning rate for the critic if the algorithm is actor critic')
    parser.add_argument('--max_episode_steps', default= 800, help= 'max number of steps per environment')
    parser.add_argument('--gamma', default= 0.999, help= 'gamma for training in the environment')
    parser.add_argument('--t_delay_theta', default= 0.9, help= 'delay for actor in case of eligibility trace')
    parser.add_argument('--t_delay_w', default= 0.9, help= 'delay for the critic in case of eligibility trace')
    parser.add_argument('--keep_patches', action= 'store_true', help= 'keep the patches for the encoder')
    parser.add_argument('--lr', default= 5e-4, help='Lr in case we need only one learning rate for our algorithm')
    parser.add_argument('--lambda_gae', default= 0.9, help='Lamda used when calculating the GAE')
    parser.add_argument('--not_normalize_advantages', action= 'store_false', help= 'normalize the advantages of each minibatch')
    parser.add_argument('--critic_eps', default= 0.3, help= 'the epsilon for clipping the critic updates' )
    parser.add_argument('--actor_eps', default= 0.3, help= 'the epsilon for clipping the actor updates' )
    parser.add_argument('--coeff_critic', default= 0.5, help= 'coefficient of the critic in the PPO general loss' )
    parser.add_argument('--coeff_entropy', default= 0.01, help= 'coefficient of the entropy in the PPO general loss' )
    parser.add_argument('--grad_clipping', action= 'store_true', help= 'do we need to clip the gradients' )
    
    #MlFlow parameters
    parser.add_argument('--track_run', action= 'store_true', help= 'track the training run with mlflow')
    parser.add_argument('--experiment_name', default= 'actor_critic_tMaze_default', help='name of experiment on mlFlow')
    parser.add_argument('--run_name', default= 'default_run', help= 'name of the run on MlFlow')
    parser.add_argument('--experiment', action= 'store_true', help= 'run a full scale MLflow experiment')

    return parser.parse_args()
    
def create_envs(args, num_envs):
    gym.envs.register(
        id='MyTMaze-v0',
        entry_point='envs.T_maze.custom_T_Maze_V0:MyTmaze'
    )
    
    envs =gym.make_vec("MyTMaze", num_envs= num_envs,  
                       max_episode_steps= args.max_episode_steps, render_mode = 'human' if args.render else None)
   

    if args.greyscale:
        envs = gym.wrappers.vector.GrayscaleObservation(envs)
    return envs
    
def launch_experiment(opt, run_dicts, seeds ,experiment_name, device, models_dict):

    from main import train
    
    create_ml_flow_experiment(experiment_name)

    model_path = os.path.abspath('trained_models')

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'mps':
            torch.mps.manual_seed(seed)
        elif device.type == 'cuda':
            torch.cuda.manual_seed(seed)
        else:
            print('not possible to assign seed')
        for run_dict in run_dicts:
            env = create_envs(opt,1)

            for key in run_dict:
                setattr(opt,key,run_dict[key])

            create_envs(opt,1)
            train(opt, env, model_path,device, models_dict)
            


def save_models(models_dict):
    
    torch.save(models_dict,'trained_models/saved_from_run.pt')



def create_ml_flow_experiment(experiment_name,uri ="file:mlruns"):
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(experiment_name)
    except MlflowException:
        mlflow.create_experiment(experiment_name)





def get_wall_states(env, pos_list, direction_list, device):
    
    states = []
    for p, d in zip(pos_list, direction_list):
        d = d * math.pi/180
    
        state, _= env.reset()
        env.unwrapped.agent.pos = p  # p = (x, 0, z)
        env.unwrapped.agent.dir = d 
        state = env.unwrapped.render_obs()
        env.render()
        state = torch.tensor(state, device= device, dtype= torch.float32)
        state = state.reshape(state.shape[2], state.shape[0], state.shape[1]) 

        states.append(state)

    states = torch.stack(states)
   
    return states


# def collect_features(env, model_path, device, pos_list, direction_list, all_layers = False):
#     encoder = load_model(model_path= model_path).eval()
#     encoder.to(device)

#     if device.type == 'mps':
#         encoder.compile(backend="aot_eager")
#     else:
#         encoder.compile()

#     for param in encoder.parameters():
#         param.requires_grad = False

#     states = get_wall_states(env, pos_list, direction_list, device)
    
#     features = encoder(states)

#     return features

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
        if max(similarities) > similarity_threshold:
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
                 alpha=0.5, beta=0.5, device='cpu', encoder=None):
        self.device = device
        self.feature_dim = feature_dim
        self.alpha = alpha  # Count-based reward coefficient
        self.beta = beta    # Temporal distance reward coefficient
        
        # Networks
        self.encoder =encoder.to(device)
        self.comparator = Comparator(feature_dim).to(device)
        
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
        print(f"Count-based reward: {count_reward.item()}")
        
        # Temporal distance reward
        temporal_reward = self._compute_temporal_distance_reward(feats)
        print(f"Temporal distance reward: {temporal_reward.item()}")
        
        # Combined intrinsic reward
        intrinsic_reward = self.alpha * count_reward + self.beta * temporal_reward
        
        # Add to memory if reward is high enough (novelty threshold)
        novelty_threshold = 0.25
        if intrinsic_reward > novelty_threshold:
            self.memory.add(obs, feats)
        
        # Store for temporal comparator training
        self._store_training_data(obs, feats, episode_step)
        
        return intrinsic_reward.item()
    
    def _compute_count_based_reward(self, features):
        """Compute count-based intrinsic reward"""
        
        
        # Find similar observations in memory
        visit_count = 0
        similarity_threshold = 0.5
        
        for stored_feat in self.memory.features:
            stored_feat = stored_feat.unsqueeze(0) if len(stored_feat.shape) == 1 else stored_feat
            similarity = F.cosine_similarity(features, stored_feat, dim=-1)
            print(f"Similarity: {similarity.item()}")
            
            if similarity.item() > similarity_threshold:
                visit_count += 10
        
        # Count-based reward: higher for less visited states
        count_reward = self.alpha / (visit_count + 1) 
        return torch.tensor(count_reward, device=self.device)
    
    def _compute_temporal_distance_reward(self, features):
        """Compute temporal distance based reward"""
        
        
        min_temporal_distance = 0.0
        
        for stored_feat in self.memory.features:
            stored_feat = stored_feat.unsqueeze(0) if len(stored_feat.shape) == 1 else stored_feat
            
            # Use comparator to get temporal correlation
            temporal_correlation = self.comparator(features, stored_feat)
            
            # Convert correlation to distance (1 - correlation)
            temporal_distance = 1.0 - temporal_correlation
            min_temporal_distance = min(min_temporal_distance, temporal_distance.item())
        
        # Reward proportional to minimum temporal distance
        temporal_reward = self.beta * min_temporal_distance
        return torch.tensor(temporal_reward, device=self.device)
    
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



    

