import torch
import torch.nn as nn

from .models import ActorModel, CriticModel, IntrinsicMotivationSystem,SpatialRepresentation

class AC_Agent(nn.Module):

    def __init__(self,num_features, num_action, activation, encoder,normalize_features = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.normalize_features = normalize_features

        self.encoder = encoder

        self.actor = ActorModel(num_features, num_action)
     
        self.critic = CriticModel(num_features, activation)

        if self.normalize_features:
            self.normalization = nn.LayerNorm(normalized_shape= num_features)

    def get_features(self, state, keep_patches = False):
        if self.encoder is not None:
            with torch.no_grad():
                x = self.encoder(state) 
                if self.normalize_features:
                    x = self.normalization(x)
        else :
            x = state
            print(state.shape)
            print(x)
        return x
    
    def get_value_from_features(self, features):
        return self.critic(features)
    
    def get_probabilities_from_features(self, features):
        return self.actor(features)
    
    def get_value_from_state(self, state):
        return self.get_value_from_features(self.get_features(state))
    
    def get_probabilities_from_state(self, state):
        return self.get_probabilities_from_features(self.get_features(state))
    

    def get_action_and_log_prob_from_features(self, features):
        if self.encoder is None:
            features=features.squeeze().squeeze().squeeze()
        probs = self.get_probabilities_from_features(features)
        dist = torch.distributions.Categorical(probs= probs)
        action = dist.sample()
        return action, dist.log_prob(action)
    
    def get_log_probs_entropy_from_features(self, features, action):
        if self.encoder is None:
            features=features.squeeze().squeeze().squeeze()
        probs = self.get_probabilities_from_features(features)
        dist = torch.distributions.Categorical(probs= probs)

        return dist.log_prob(action), dist.entropy()
    
    def get_action_and_log_prob_dist_from_features(self, features):
        if self.encoder is None:
            features=features.squeeze().squeeze().squeeze()
        probs = self.get_probabilities_from_features(features)
        dist = torch.distributions.Categorical(probs= probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist






class AC_Agent_With_Buffer(AC_Agent): 
     def __init__(self,num_features, num_action, activation, encoder, normalize= False):
        super().__init__(num_action=num_action, num_features=num_features, activation=activation, encoder=encoder, normalize_features=normalize)
        self.motivator = IntrinsicMotivationSystem(num_features, encoder=encoder)

     def compute_intrinsic_rewards(self,obs,feats):
         return self.motivator.compute_intrinsic_reward(obs, feats)
     
     def train_temporal_comparator(self):
         self.motivator.train_temporal_comparator()

class AC_Agent_With_Spatial_Representations(AC_Agent):
    def __init__(self, num_features, num_action, activation, encoder, normalize):
        super().__init__(num_features=num_features, num_action=num_action,activation=activation,encoder=encoder,normalize_features=normalize)
        self.encoder = encoder
        self.spatial_rep = SpatialRepresentation(
            input_dim=num_features,
        )
        
        # Actor now takes spatial features
        self.actor = ActorModel(num_features=num_features+self.spatial_rep.hd_cell_dim + self.spatial_rep.place_cell_dim,num_actions= num_action)  # place + hd cells
        
        # Critic also uses spatial features
        self.critic = CriticModel(num_features=num_features+self.spatial_rep.hd_cell_dim + self.spatial_rep.place_cell_dim,activation= activation)
        
    def get_features(self, state, keep_patches=False):
        visual_features = self.encoder(state, keep_patches)
        spatial_features = self.spatial_rep(visual_features)
        
        return torch.cat((visual_features,spatial_features), dim=-1)
    
    def reset_spatial_memory(self):
        self.spatial_rep.reset()
        


    

    
    
    
    
    
        

    