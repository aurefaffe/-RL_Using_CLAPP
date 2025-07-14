import seaborn as sns
import math
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from utils.load_standalone_model import load_model
import os
import gymnasium as gym
from utils.utils import parsing, create_envs
import numpy


class TmazeDiscretizer:
    def __init__(self, env, encoder=None):
        self.env1 = self._unwrap_env(env)
        self.env = env
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = encoder
        self.featureslist = []
        
        # Positions discrétisées basées sur ton code
        self.discrete_positions = self._generate_discrete_positions()
        
    def _generate_discrete_positions(self):
        """Génère les positions discrétisées du T-maze"""
        positions = []
        
        # Corridor principal (room1) - murs du haut et du bas
        for x in range(3):
            positions.append([1.37*(2*x+1)-0.22, 1.37, -1.37])  # Mur du bas
            positions.append([1.37*(2*x+1)-0.22, 1.37, 1.37])   # Mur du haut
            
        # Bras gauche et droit (room2) - mur du fond
        for x in range(5):
            positions.append([10.74, 1.37, 1.37*(2*x+1)-6.85])
            
        # Jonction entre room1 et room2
        for x in [0, 1, 3, 4]:  # Exclut la position centrale (x=2)
            positions.append([8, 1.37, 1.37*(2*x+1)-6.85])
            
        # Coins des bras
        positions.append([9.37, 1.37, -6.85])  # Coin bras gauche
        positions.append([9.37, 1.37, 6.85])   # Coin bras droit
        
        return positions
    
    def get_grid_positions(self, resolution=0.5):
        """Génère une grille de positions dans l'espace navigable"""
        grid_positions = []
        
        # Room1: corridor principal
        x_range = np.arange(-0.22, 8, resolution)
        z_range = np.arange(-1.37, 1.37, resolution)
        
        for x in x_range:
            for z in z_range:
                grid_positions.append([x, 1.37, z])
        
        # Room2: bras du T
        x_range = np.arange(8, 10.74, resolution)
        z_min = -6.85 if self.env.left_arm else -1.37
        z_max = 6.85 if self.env.right_arm else 1.37
        z_range = np.arange(z_min, z_max, resolution)
        
        for x in x_range:
            for z in z_range:
                grid_positions.append([x, 1.37, z])
                
        return grid_positions
    
    def extract_features(self, obs):
        """Extrait les caractéristiques de l'observation"""
        if self.encoder is not None:
            print(obs)
            obs = obs[0]
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            obs_tensor = obs_tensor.reshape(obs_tensor.shape[2],1,obs_tensor.shape[1], obs_tensor.shape[0])
            with torch.no_grad():
                features = self.encoder(obs_tensor)
            return features.cpu().numpy()
    
    def extract_features_from_all_positions(self, positions=None, orientations=None):
        """
        Extrait les features de toutes les positions et orientations
        
        Args:
            positions: Liste des positions à tester (par défaut self.discrete_positions)
            orientations: Liste des orientations en degrés (par défaut [0, 45, 90, 135, 180, 225, 270, 315])
        """
        if positions is None:
            positions = self.discrete_positions
        
        if orientations is None:
            orientations = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 orientations
        
        self.featureslist = []
        position_orientation_pairs = []
        
        for pos_idx, pos in enumerate(positions):
            for orient in orientations:
                # try:
                    # Reset et placer l'agent
                    wsh = self.env1.reset()
                   
                    features = self.extract_features(wsh)
                    print(f"Position {pos_idx}, Orientation {orient}°: Features shape {features.shape}")
                    self.env1.agent.pos=pos
                    self.env1.agent.dir = orient 
                    
                    features = features.flatten()
                    
                    self.featureslist.append(features)
                    position_orientation_pairs.append((pos_idx, orient))
                    
                # except Exception as e:
                #     print(f"Erreur à la position {pos} avec orientation {orient}: {e}")
                #     continue
        
        self.featureslist = np.array(self.featureslist)
        self.position_orientation_pairs = position_orientation_pairs
        
        return self.featureslist
    
    def compute_similarity_matrix(self, features=None):
        """
        Calcule la matrice de similarité cosinus entre toutes les features
        
        Args:
            features: Features à utiliser (par défaut self.featureslist)
        
        Returns:
            Matrice de similarité cosinus
        """
        if features is None:
            if len(self.featureslist) == 0:
                raise ValueError("Aucune feature extraite. Appelez d'abord extract_features_from_all_positions()")
            features = self.featureslist
        
        # Calculer la similarité cosinus
        similarity_matrix = cosine_similarity(features)
        
        return similarity_matrix
    
    def visualize_similarity_matrix(self, similarity_matrix=None, positions_list=None, 
                                  show_orientations=False, save_path=None):
        """
        Visualise la matrice de similarité
        
        Args:
            similarity_matrix: Matrice de similarité (calculée automatiquement si None)
            positions_list: Liste des positions (par défaut self.discrete_positions)
            show_orientations: Si True, affiche les orientations dans les labels
            save_path: Chemin pour sauvegarder l'image (optionnel)
        """
        if similarity_matrix is None:
            similarity_matrix = self.compute_similarity_matrix()
        
        if positions_list is None:
            positions_list = self.discrete_positions
        
        plt.figure(figsize=(15, 12))
        
        # Créer des labels
        if show_orientations and hasattr(self, 'position_orientation_pairs'):
            labels = []
            for pos_idx, orient in self.position_orientation_pairs:
                pos = positions_list[pos_idx]
                labels.append(f"P{pos_idx}_{orient}°\n({pos[0]:.1f},{pos[2]:.1f})")
        else:
            labels = [f"P{i}\n({pos[0]:.1f},{pos[2]:.1f})" for i, pos in enumerate(positions_list)]
        
        # Adapter les labels à la taille de la matrice
        if len(labels) != similarity_matrix.shape[0]:
            labels = [f"F{i}" for i in range(similarity_matrix.shape[0])]
        
        sns.heatmap(similarity_matrix, 
                    annot=False,  # Désactiver les annotations pour plus de clarté
                    cmap='viridis',
                    xticklabels=labels if len(labels) <= 50 else False,  # Cacher les labels si trop nombreux
                    yticklabels=labels if len(labels) <= 50 else False,
                    cbar_kws={'label': 'Similarité cosinus'})
        
        plt.title('Matrice de Similarité CLIP - T-maze Discrétisé')
        plt.xlabel('Positions/Orientations')
        plt.ylabel('Positions/Orientations')
        
        if len(labels) <= 50:
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_position_similarity_summary(self, positions_list=None):
        """
        Résume la similarité entre positions (moyenne sur les orientations)
        
        Returns:
            Matrice de similarité entre positions (moyennée sur les orientations)
        """
        if positions_list is None:
            positions_list = self.discrete_positions
        
        if not hasattr(self, 'position_orientation_pairs'):
            raise ValueError("Pas de données d'orientation disponibles")
        
        n_positions = len(positions_list)
        position_similarity = np.zeros((n_positions, n_positions))
        
        # Regrouper par position
        for i in range(n_positions):
            for j in range(n_positions):
                similarities = []
                for idx1, (pos1, orient1) in enumerate(self.position_orientation_pairs):
                    for idx2, (pos2, orient2) in enumerate(self.position_orientation_pairs):
                        if pos1 == i and pos2 == j:
                            sim_matrix = self.compute_similarity_matrix()
                            similarities.append(sim_matrix[idx1, idx2])
                
                if similarities:
                    position_similarity[i, j] = np.mean(similarities)
        
        return position_similarity
    def _unwrap_env(self, env):
        """
        Unwrap l'environnement pour accéder à l'environnement de base
        """
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env

if __name__ == '__main__':
    model_path = os.path.abspath('trained_models')
    encoder = load_model(model_path=model_path)
    gym.envs.register(
        id='MyTMaze-v0',
        entry_point='envs.T_maze.custom_T_Maze_V0:MyTmaze'
    )
    
    envs =gym.make("MyTMaze")
    envs = gym.wrappers.GrayscaleObservation(envs)
   
    
    
    # Create discretizer
    TmazeforMatrix = TmazeDiscretizer(env=envs, encoder=encoder)
    
    # Extract features and compute similarity
    features = TmazeforMatrix.extract_features_from_all_positions()
    matrice = TmazeforMatrix.compute_similarity_matrix(features=features)
    
    # Visualize
    TmazeforMatrix.visualize_similarity_matrix(similarity_matrix=matrice)
    