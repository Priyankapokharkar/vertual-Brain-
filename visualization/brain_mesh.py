"""
Brain Mesh Generation and Processing
Handles 3D brain geometry for visualization
"""
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class BrainMesh:
    """
    3D brain mesh representation and processing
    Generates mesh data for Three.js rendering
    """
    
    def __init__(self):
        """Initialize brain mesh"""
        self.vertices = None
        self.faces = None
        self.normals = None
        self.regions = {}
        self.colors = None
        
    def load_obj(self, file_path: Path) -> None:
        """
        Load brain mesh from OBJ file
        
        Args:
            file_path: Path to OBJ file
        """
        vertices = []
        faces = []
        normals = []
        
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                if parts[0] == 'v':  # Vertex
                    vertices.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'vn':  # Normal
                    normals.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'f':  # Face
                    # Parse face indices (can be v, v/vt, or v/vt/vn)
                    face = []
                    for vertex_info in parts[1:]:
                        face.append(int(vertex_info.split('/')[0]) - 1)  # OBJ is 1-indexed
                    faces.append(face)
        
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)
        if normals:
            self.normals = np.array(normals)
        else:
            self.normals = self._calculate_normals()
    
    def generate_synthetic_brain(self, resolution: int = 50) -> None:
        """
        Generate synthetic brain-shaped mesh
        
        Args:
            resolution: Mesh resolution
        """
        # Generate brain-like ellipsoid
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        
        vertices = []
        faces = []
        
        for i, v_angle in enumerate(v):
            for j, u_angle in enumerate(u):
                # Ellipsoid with brain-like proportions
                x = 0.8 * np.sin(v_angle) * np.cos(u_angle)
                y = 0.8 * np.sin(v_angle) * np.sin(u_angle)
                z = 1.2 * np.cos(v_angle)
                
                # Add some deformation for realism
                deform = 0.05 * (np.sin(5 * u_angle) * np.cos(3 * v_angle))
                r = 1.0 + deform
                
                vertices.append([r * x, r * y, r * z])
                
                # Create faces (triangles)
                if i < len(v) - 1 and j < len(u) - 1:
                    idx = i * len(u) + j
                    # Two triangles per quad
                    faces.append([idx, idx + 1, idx + len(u)])
                    faces.append([idx + 1, idx + len(u) + 1, idx + len(u)])
        
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)
        self.normals = self._calculate_normals()
        
    def _calculate_normals(self) -> np.ndarray:
        """Calculate vertex normals from face data"""
        normals = np.zeros_like(self.vertices)
        
        for face in self.faces:
            # Get vertices of the triangle
            v0, v1, v2 = self.vertices[face[:3]]
            
            # Calculate face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            
            # Accumulate normals for each vertex
            for idx in face[:3]:
                normals[idx] += normal
        
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-10)
        
        return normals
    
    def add_region(self, region_name: str, vertex_indices: List[int], color: Tuple[float, float, float]) -> None:
        """
        Define a brain region
        
        Args:
            region_name: Name of the region
            vertex_indices: Indices of vertices belonging to this region
            color: RGB color tuple (0-1 range)
        """
        self.regions[region_name] = {
            'vertices': vertex_indices,
            'color': color
        }
    
    def parcellate(self, num_regions: int = 90) -> Dict:
        """
        Parcellate brain into regions (simplified clustering)
        
        Args:
            num_regions: Number of regions
        
        Returns:
            Dictionary mapping region IDs to vertex indices
        """
        # Simple k-means-like parcellation based on position
        from scipy.cluster.vq import kmeans2
        
        # Cluster vertices
        centroids, labels = kmeans2(self.vertices, num_regions, minit='points')
        
        # Generate colors for each region
        colors = []
        for i in range(num_regions):
            hue = i / num_regions
            colors.append(self._hsv_to_rgb(hue, 0.7, 0.9))
        
        parcellation = {}
        for region_id in range(num_regions):
            vertex_indices = np.where(labels == region_id)[0].tolist()
            parcellation[f"region_{region_id}"] = {
                'vertices': vertex_indices,
                'centroid': centroids[region_id].tolist(),
                'color': colors[region_id]
            }
        
        return parcellation
    
    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Convert HSV to RGB"""
        import colorsys
        return colorsys.hsv_to_rgb(h, s, v)
    
    def highlight_region(self, region_name: str, intensity: float = 1.0) -> np.ndarray:
        """
        Generate color array with highlighted region
        
        Args:
            region_name: Name of region to highlight
            intensity: Highlight intensity (0-1)
        
        Returns:
            Color array for all vertices
        """
        if self.colors is None:
            # Default gray color
            self.colors = np.ones((len(self.vertices), 3)) * 0.7
        
        colors = self.colors.copy()
        
        if region_name in self.regions:
            vertex_indices = self.regions[region_name]['vertices']
            highlight_color = np.array(self.regions[region_name]['color'])
            colors[vertex_indices] = highlight_color * intensity
        
        return colors
    
    def export_to_threejs(self) -> Dict:
        """
        Export mesh data in Three.js compatible format
        
        Returns:
            Dictionary with vertices, faces, and normals
        """
        data = {
            'vertices': self.vertices.flatten().tolist(),
            'faces': self.faces.flatten().tolist(),
            'normals': self.normals.flatten().tolist() if self.normals is not None else [],
            'num_vertices': len(self.vertices),
            'num_faces': len(self.faces)
        }
        
        if self.regions:
            data['regions'] = {
                name: {
                    'vertices': region['vertices'],
                    'color': region['color']
                }
                for name, region in self.regions.items()
            }
        
        return data
    
    def export_to_obj(self, file_path: Path) -> None:
        """
        Export mesh to OBJ file
        
        Args:
            file_path: Output file path
        """
        with open(file_path, 'w') as f:
            # Write header
            f.write("# Brain Mesh\n")
            f.write(f"# Vertices: {len(self.vertices)}\n")
            f.write(f"# Faces: {len(self.faces)}\n\n")
            
            # Write vertices
            for v in self.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            # Write normals
            if self.normals is not None:
                for n in self.normals:
                    f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
            
            # Write faces (OBJ is 1-indexed)
            for face in self.faces:
                if self.normals is not None:
                    f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
                else:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    def simplify_mesh(self, target_faces: int) -> None:
        """
        Simplify mesh by reducing number of faces
        This is a simplified version - production would use proper decimation
        
        Args:
            target_faces: Target number of faces
        """
        if len(self.faces) <= target_faces:
            return
        
        # Simple random sampling for demonstration
        # In production, use proper mesh decimation algorithms
        keep_ratio = target_faces / len(self.faces)
        keep_indices = np.random.choice(len(self.faces), target_faces, replace=False)
        
        self.faces = self.faces[keep_indices]
        self.normals = self._calculate_normals()
    
    def get_bounding_box(self) -> Dict:
        """Get bounding box of the mesh"""
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        
        return {
            'min': min_coords.tolist(),
            'max': max_coords.tolist(),
            'center': center.tolist(),
            'size': size.tolist()
        }
    
    def apply_disease_deformation(self, disease_type: str, severity: float = 0.5, focal_point: Optional[np.ndarray] = None) -> Dict:
        """
        Apply deformation to mesh based on disease type
        
        Args:
            disease_type: Type of disease
            severity: Deformation severity (0-1)
            focal_point: Optional coordinate for focal diseases
            
        Returns:
            Metadata about the deformation (center, radius, impacted_regions)
        """
        metadata = {'center': None, 'radius': 0}
        
        if disease_type == 'alzheimer':
            # Simulate atrophy (shrinkage) in temporal regions
            for i, vertex in enumerate(self.vertices):
                if vertex[1] < 0:  # Lower regions
                    self.vertices[i] *= (1.0 - 0.1 * severity)
        
        elif disease_type == 'tumor':
            # Simulate localized mass effect (bulge)
            tumor_center = focal_point if focal_point is not None else np.array([0.4, 0.3, 0.4])
            tumor_radius = 0.4 * severity
            
            metadata['center'] = tumor_center.tolist()
            metadata['radius'] = tumor_radius
            
            for i, vertex in enumerate(self.vertices):
                dist = np.linalg.norm(vertex - tumor_center)
                if dist < tumor_radius:
                    direction = vertex - tumor_center
                    norm_dir = direction / (np.linalg.norm(direction) + 1e-10)
                    push = (1.0 - (dist / tumor_radius)) * 0.2 * severity
                    self.vertices[i] += norm_dir * push
                    
        elif disease_type == 'stroke':
            # Simulate localized damage
            lesion_center = focal_point if focal_point is not None else np.array([0.5, 0.5, 0.0])
            lesion_radius = 0.3 * severity
            
            metadata['center'] = lesion_center.tolist()
            metadata['radius'] = lesion_radius
            
            for i, vertex in enumerate(self.vertices):
                dist = np.linalg.norm(vertex - lesion_center)
                if dist < lesion_radius:
                    self.vertices[i] += np.random.randn(3) * 0.05 * severity
        
        # Recalculate normals after deformation
        self.normals = self._calculate_normals()
        return metadata
