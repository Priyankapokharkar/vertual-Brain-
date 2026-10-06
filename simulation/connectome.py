"""
Structural Connectome Module
Represents and processes brain connectivity networks
"""
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class Connectome:
    """
    Structural connectome representation
    Models brain regions and their white-matter connections
    """
    
    def __init__(self, num_regions: int = 90):
        """
        Initialize connectome
        
        Args:
            num_regions: Number of brain regions (default: 90 for AAL atlas)
        """
        self.num_regions = num_regions
        self.connectivity_matrix = None
        self.region_names = []
        self.region_positions = None
        self.weights = None
        self.distances = None
        
    def load_from_file(self, file_path: Path) -> None:
        """Load connectome data from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.connectivity_matrix = np.array(data['connectivity_matrix'])
        self.region_names = data.get('region_names', [f"Region_{i}" for i in range(self.num_regions)])
        self.region_positions = np.array(data.get('region_positions', self._generate_default_positions()))
        self.weights = np.array(data.get('weights', self.connectivity_matrix))
        
        # Calculate distances if not provided
        if 'distances' in data:
            self.distances = np.array(data['distances'])
        else:
            self.distances = self._calculate_distances()
    
    def generate_synthetic(self, connectivity_density: float = 0.3) -> None:
        """
        Generate synthetic connectome for demonstration
        
        Args:
            connectivity_density: Proportion of possible connections
        """
        # Generate random connectivity matrix
        random_matrix = np.random.rand(self.num_regions, self.num_regions)
        
        # Make it symmetric (undirected graph)
        random_matrix = (random_matrix + random_matrix.T) / 2
        
        # Apply density threshold
        threshold = np.percentile(random_matrix, (1 - connectivity_density) * 100)
        self.connectivity_matrix = (random_matrix > threshold).astype(float)
        
        # Remove self-connections
        np.fill_diagonal(self.connectivity_matrix, 0)
        
        # Generate weights (connection strengths)
        self.weights = random_matrix * self.connectivity_matrix
        self.weights = self.weights / self.weights.max()  # Normalize
        
        # Generate region names
        self.region_names = [f"Region_{i}" for i in range(self.num_regions)]
        
        # Generate 3D positions for regions
        self.region_positions = self._generate_default_positions()
        
        # Calculate distances
        self.distances = self._calculate_distances()
    
    def _generate_default_positions(self) -> np.ndarray:
        """Generate default 3D positions for brain regions"""
        # Create positions in a brain-like ellipsoid
        positions = []
        
        for i in range(self.num_regions):
            # Distribute points in 3D space
            theta = 2 * np.pi * i / self.num_regions
            phi = np.arccos(2 * (i / self.num_regions) - 1)
            
            # Scale by brain-like ellipsoid
            x = 0.8 * np.sin(phi) * np.cos(theta)
            y = 0.8 * np.sin(phi) * np.sin(theta)
            z = 1.2 * np.cos(phi)
            
            positions.append([x, y, z])
        
        return np.array(positions)
    
    def _calculate_distances(self) -> np.ndarray:
        """Calculate Euclidean distances between regions"""
        distances = np.zeros((self.num_regions, self.num_regions))
        
        for i in range(self.num_regions):
            for j in range(self.num_regions):
                distances[i, j] = np.linalg.norm(
                    self.region_positions[i] - self.region_positions[j]
                )
        
        return distances
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get binary adjacency matrix"""
        return self.connectivity_matrix
    
    def get_weighted_matrix(self) -> np.ndarray:
        """Get weighted connectivity matrix"""
        return self.weights
    
    def get_degree_distribution(self) -> Dict:
        """Calculate degree distribution"""
        degrees = np.sum(self.connectivity_matrix, axis=1)
        
        return {
            'degrees': degrees.tolist(),
            'mean_degree': float(np.mean(degrees)),
            'std_degree': float(np.std(degrees)),
            'max_degree': int(np.max(degrees)),
            'min_degree': int(np.min(degrees))
        }
    
    def get_network_metrics(self) -> Dict:
        """Calculate network topology metrics"""
        if not NETWORKX_AVAILABLE:
            return self._simple_network_metrics()
        
        # Create NetworkX graph
        G = nx.from_numpy_array(self.connectivity_matrix)
        
        metrics = {
            'num_nodes': self.num_regions,
            'num_edges': int(np.sum(self.connectivity_matrix) / 2),
            'density': float(nx.density(G)),
            'average_clustering': float(nx.average_clustering(G)),
            'transitivity': float(nx.transitivity(G))
        }
        
        # Calculate additional metrics if graph is connected
        if nx.is_connected(G):
            metrics['average_shortest_path'] = float(nx.average_shortest_path_length(G))
            metrics['diameter'] = int(nx.diameter(G))
            metrics['global_efficiency'] = float(nx.global_efficiency(G))
        
        return metrics
    
    def _simple_network_metrics(self) -> Dict:
        """Simple network metrics without NetworkX"""
        num_edges = int(np.sum(self.connectivity_matrix) / 2)
        max_edges = self.num_regions * (self.num_regions - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0
        
        return {
            'num_nodes': self.num_regions,
            'num_edges': num_edges,
            'density': float(density),
            'average_degree': float(np.mean(np.sum(self.connectivity_matrix, axis=1)))
        }
    
    def get_hub_regions(self, top_k: int = 10) -> List[Dict]:
        """
        Identify hub regions (highly connected nodes)
        
        Args:
            top_k: Number of top hubs to return
        
        Returns:
            List of hub regions with their connectivity info
        """
        degrees = np.sum(self.connectivity_matrix, axis=1)
        strengths = np.sum(self.weights, axis=1)
        
        # Get top hubs
        hub_indices = np.argsort(degrees)[-top_k:][::-1]
        
        hubs = []
        for idx in hub_indices:
            hubs.append({
                'region_id': int(idx),
                'region_name': self.region_names[idx],
                'degree': int(degrees[idx]),
                'strength': float(strengths[idx]),
                'position': self.region_positions[idx].tolist()
            })
        
        return hubs
    
    def get_modular_structure(self) -> Dict:
        """Detect modular/community structure"""
        if not NETWORKX_AVAILABLE:
            return {'modules': 'NetworkX required for community detection'}
        
        from networkx.algorithms import community
        
        G = nx.from_numpy_array(self.connectivity_matrix)
        communities = community.greedy_modularity_communities(G)
        
        module_assignment = np.zeros(self.num_regions, dtype=int)
        for module_id, nodes in enumerate(communities):
            for node in nodes:
                module_assignment[node] = module_id
        
        return {
            'num_modules': len(communities),
            'module_assignment': module_assignment.tolist(),
            'modularity': float(community.modularity(G, communities))
        }
    
    def simulate_lesion(self, region_ids: List[int]) -> Dict:
        """
        Simulate lesion (removal of regions) and assess impact
        
        Args:
            region_ids: List of region indices to lesion
        
        Returns:
            Impact metrics
        """
        # Create lesioned connectivity matrix
        lesioned_matrix = self.connectivity_matrix.copy()
        lesioned_matrix[region_ids, :] = 0
        lesioned_matrix[:, region_ids] = 0
        
        # Calculate impact
        original_edges = int(np.sum(self.connectivity_matrix) / 2)
        lesioned_edges = int(np.sum(lesioned_matrix) / 2)
        edges_lost = original_edges - lesioned_edges
        
        return {
            'lesioned_regions': [self.region_names[i] for i in region_ids],
            'original_edges': original_edges,
            'remaining_edges': lesioned_edges,
            'edges_lost': edges_lost,
            'connectivity_loss_percent': float(edges_lost / original_edges * 100) if original_edges > 0 else 0
        }

    def apply_spatial_lesion(self, center: np.ndarray, radius: float) -> List[int]:
        """
        Identify regions within a spatial radius and simulate a lesion
        
        Args:
            center: 3D coordinates of lesion center
            radius: Radius of effect
            
        Returns:
            List of affected region indices
        """
        affected_regions = []
        center = np.array(center)
        
        for i, pos in enumerate(self.region_positions):
            dist = np.linalg.norm(pos - center)
            if dist < radius:
                affected_regions.append(i)
        
        if affected_regions:
            # Modify the connectivity matrix and weights
            self.connectivity_matrix[affected_regions, :] = 0
            self.connectivity_matrix[:, affected_regions] = 0
            self.weights[affected_regions, :] = 0
            self.weights[:, affected_regions] = 0
            
        return affected_regions
    
    def export_to_json(self, file_path: Path) -> None:
        """Export connectome data to JSON"""
        data = {
            'num_regions': self.num_regions,
            'connectivity_matrix': self.connectivity_matrix.tolist(),
            'weights': self.weights.tolist(),
            'region_names': self.region_names,
            'region_positions': self.region_positions.tolist(),
            'distances': self.distances.tolist(),
            'network_metrics': self.get_network_metrics()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_subnetwork(self, region_ids: List[int]) -> 'Connectome':
        """Extract subnetwork containing specified regions"""
        sub_connectome = Connectome(num_regions=len(region_ids))
        
        # Extract submatrix
        idx = np.ix_(region_ids, region_ids)
        sub_connectome.connectivity_matrix = self.connectivity_matrix[idx]
        sub_connectome.weights = self.weights[idx]
        sub_connectome.region_positions = self.region_positions[region_ids]
        sub_connectome.region_names = [self.region_names[i] for i in region_ids]
        sub_connectome.distances = self._calculate_distances()
        
        return sub_connectome
