"""
Blood Vessel Network Modeling
Generates and processes vascular network for visualization
"""
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class VesselNetwork:
    """
    Blood vessel network representation
    Models arterial and venous structures
    """
    
    def __init__(self):
        """Initialize vessel network"""
        self.vessels = []
        self.vessel_tree = None
        self.flow_rates = []
        
    def generate_synthetic_network(self, num_vessels: int = 100, 
                                   network_type: str = 'mixed') -> None:
        """
        Generate synthetic vascular network
        
        Args:
            num_vessels: Number of vessel segments
            network_type: 'arterial', 'venous', or 'mixed'
        """
        self.vessels = []
        
        # Major vessels starting points
        if network_type in ['arterial', 'mixed']:
            # Arterial tree (from circle of Willis)
            self._generate_arterial_tree(num_vessels // 2 if network_type == 'mixed' else num_vessels)
        
        if network_type in ['venous', 'mixed']:
            # Venous system
            self._generate_venous_system(num_vessels // 2 if network_type == 'mixed' else num_vessels)
    
    def _generate_arterial_tree(self, num_vessels: int) -> None:
        """Generate arterial tree structure"""
        # Start from base (circle of Willis)
        base_points = [
            np.array([0.0, 0.0, -0.5]),  # Basilar artery
            np.array([0.3, 0.0, -0.3]),  # Right ICA
            np.array([-0.3, 0.0, -0.3]), # Left ICA
        ]
        
        vessel_id = len(self.vessels)
        
        for base_point in base_points:
            self._grow_vessel_branch(
                start_point=base_point,
                direction=np.array([0.0, 0.0, 1.0]),  # Upward
                depth=0,
                max_depth=5,
                vessel_id=vessel_id,
                vessel_type='artery',
                parent_radius=0.03
            )
            vessel_id = len(self.vessels)
    
    def _generate_venous_system(self, num_vessels: int) -> None:
        """Generate venous drainage system"""
        # Major venous sinuses
        sinus_points = [
            np.array([0.0, 0.0, 1.0]),   # Superior sagittal sinus
            np.array([0.5, 0.5, 0.5]),   # Right transverse sinus
            np.array([-0.5, 0.5, 0.5]),  # Left transverse sinus
        ]
        
        vessel_id = len(self.vessels)
        
        for sinus_point in sinus_points:
            self._grow_vessel_branch(
                start_point=sinus_point,
                direction=np.array([0.0, 0.0, -1.0]),  # Downward drainage
                depth=0,
                max_depth=4,
                vessel_id=vessel_id,
                vessel_type='vein',
                parent_radius=0.025
            )
            vessel_id = len(self.vessels)
    
    def _grow_vessel_branch(self, start_point: np.ndarray, direction: np.ndarray,
                           depth: int, max_depth: int, vessel_id: int,
                           vessel_type: str, parent_radius: float) -> None:
        """
        Recursively grow vessel branches
        
        Args:
            start_point: Starting point of vessel
            direction: Growth direction
            depth: Current branch depth
            max_depth: Maximum branching depth
            vessel_id: Vessel identifier
            vessel_type: 'artery' or 'vein'
            parent_radius: Radius of parent vessel
        """
        if depth >= max_depth or len(self.vessels) > 200:
            return
        
        # Calculate segment length and radius
        segment_length = 0.2 * (0.7 ** depth) + np.random.rand() * 0.1
        radius = parent_radius * (0.8 ** depth)
        
        # Normalize direction and add some randomness
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        direction += np.random.randn(3) * 0.2
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Calculate end point
        end_point = start_point + direction * segment_length
        
        # Create vessel segment
        vessel = {
            'id': len(self.vessels),
            'type': vessel_type,
            'start': start_point.tolist(),
            'end': end_point.tolist(),
            'radius': radius,
            'length': segment_length,
            'depth': depth,
            'parent_id': vessel_id if depth > 0 else None
        }
        
        self.vessels.append(vessel)
        current_id = len(self.vessels) - 1
        
        # Branching probability decreases with depth
        branch_prob = 0.7 * (0.8 ** depth)
        
        if np.random.rand() < branch_prob:
            # Create bifurcation
            num_branches = 2 if np.random.rand() < 0.8 else 3
            
            for _ in range(num_branches):
                # Generate branch direction
                tangent = np.random.randn(3)
                tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
                
                # Mix with parent direction
                branch_direction = 0.7 * direction + 0.3 * tangent
                
                # Recurse
                self._grow_vessel_branch(
                    end_point, branch_direction, depth + 1, max_depth,
                    current_id, vessel_type, radius
                )
    
    def calculate_flow(self) -> None:
        """Calculate blood flow rates through vessels using simplified model"""
        self.flow_rates = []
        
        for vessel in self.vessels:
            # Simplified Poiseuille flow: Q ∝ r^4
            radius = vessel['radius']
            length = vessel['length']
            
            # Flow rate (arbitrary units)
            if vessel['type'] == 'artery':
                pressure_gradient = 100  # mmHg
            else:
                pressure_gradient = 10   # Lower venous pressure
            
            # Poiseuille's law (simplified)
            flow = (np.pi * radius**4 * pressure_gradient) / (8 * 0.004 * length)  # 0.004 is blood viscosity
            
            self.flow_rates.append(flow)
            vessel['flow_rate'] = flow
    
    def detect_anomalies(self) -> List[Dict]:
        """
        Detect flow anomalies that might indicate blockages
        
        Returns:
            List of vessels with anomalies
        """
        if not self.flow_rates:
            self.calculate_flow()
        
        anomalies = []
        mean_flow = np.mean(self.flow_rates)
        std_flow = np.std(self.flow_rates)
        
        for i, vessel in enumerate(self.vessels):
            flow = self.flow_rates[i]
            
            # Detect significantly reduced flow
            if flow < mean_flow - 2 * std_flow:
                severity = min(1.0, (mean_flow - flow) / (mean_flow + 1e-10))
                
                anomalies.append({
                    'vessel_id': vessel['id'],
                    'vessel_type': vessel['type'],
                    'location': vessel['start'],
                    'severity': severity,
                    'flow_rate': flow,
                    'expected_flow': mean_flow,
                    'reason': 'Reduced flow'
                })
        
        return anomalies
    
    def add_blockage(self, vessel_id: int, severity: float) -> None:
        """
        Simulate a blockage in a specific vessel
        
        Args:
            vessel_id: ID of vessel to block
            severity: Blockage severity (0-1, where 1 is complete occlusion)
        """
        if vessel_id < len(self.vessels):
            vessel = self.vessels[vessel_id]
            vessel['blockage_severity'] = severity
            vessel['stenosis'] = severity > 0.5
            
            # Reduce effective radius
            vessel['effective_radius'] = vessel['radius'] * (1 - severity)
            
            # Recalculate flow
            self.calculate_flow()

    def apply_spatial_damage(self, center: np.ndarray, radius: float, max_severity: float = 0.8) -> List[int]:
        """
        Apply damage to all vessels within a spatial radius (e.g., tumor impact)
        
        Args:
            center: 3D coordinates of damage center
            radius: Radius of effect
            max_severity: Maximum blockage severity
            
        Returns:
            List of affected vessel IDs
        """
        affected_ids = []
        center = np.array(center)
        
        for i, vessel in enumerate(self.vessels):
            v_start = np.array(vessel['start'])
            v_end = np.array(vessel['end'])
            
            # Check distance to vessel segment (clamping to segment)
            dist = self._dist_point_to_segment(center, v_start, v_end)
            
            if dist < radius:
                # Severity decreases with distance from center
                severity = max_severity * (1.0 - (dist / radius))
                vessel['blockage_severity'] = max(vessel.get('blockage_severity', 0), severity)
                vessel['effective_radius'] = vessel['radius'] * (1 - vessel['blockage_severity'])
                affected_ids.append(i)
        
        if affected_ids:
            self.calculate_flow()
            
        return affected_ids

    def _dist_point_to_segment(self, p, a, b):
        """Standard point-to-segment distance calculation"""
        pa, ba = p - a, b - a
        h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0, 1)
        return np.linalg.norm(pa - ba * h)
    
    def get_vessel_path(self, vessel_id: int) -> List[np.ndarray]:
        """
        Get smooth path points for a vessel (for rendering)
        
        Args:
            vessel_id: Vessel ID
        
        Returns:
            List of 3D points along the vessel
        """
        vessel = self.vessels[vessel_id]
        start = np.array(vessel['start'])
        end = np.array(vessel['end'])
        
        # Generate smooth curve (simple linear interpolation)
        num_points = 10
        points = []
        
        for t in np.linspace(0, 1, num_points):
            point = start * (1 - t) + end * t
            # Add slight curvature
            point += np.random.randn(3) * 0.01 * vessel['radius']
            points.append(point)
        
        return points
    
    def export_to_threejs(self) -> Dict:
        """
        Export vessel network in Three.js compatible format
        
        Returns:
            Dictionary with vessel data
        """
        vessels_data = []
        
        for vessel in self.vessels:
            vessel_data = {
                'id': vessel['id'],
                'type': vessel['type'],
                'start': vessel['start'],
                'end': vessel['end'],
                'radius': vessel['radius'],
                'flow_rate': vessel.get('flow_rate', 0),
                'blockage': vessel.get('blockage_severity', 0),
                'color': self._get_vessel_color(vessel)
            }
            vessels_data.append(vessel_data)
        
        return {
            'vessels': vessels_data,
            'num_vessels': len(self.vessels),
            'vessel_types': list(set(v['type'] for v in self.vessels))
        }
    
    def _get_vessel_color(self, vessel: Dict) -> List[float]:
        """Get color for vessel based on type and flow"""
        if vessel['type'] == 'artery':
            # Red for arteries (oxygenated blood)
            base_color = [1.0, 0.1, 0.1]
        else:
            # Blue for veins (deoxygenated blood)
            base_color = [0.1, 0.1, 1.0]
        
        # Adjust based on blockage
        if 'blockage_severity' in vessel:
            severity = vessel['blockage_severity']
            # Darken if blocked
            base_color = [c * (1 - 0.5 * severity) for c in base_color]
        
        return base_color
    
    def export_to_json(self, file_path: Path) -> None:
        """Export vessel network to JSON"""
        data = self.export_to_threejs()
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_network_statistics(self) -> Dict:
        """Get statistics about the vascular network"""
        if not self.vessels:
            return {}
        
        arterial_count = sum(1 for v in self.vessels if v['type'] == 'artery')
        venous_count = len(self.vessels) - arterial_count
        
        total_length = sum(v['length'] for v in self.vessels)
        avg_radius = np.mean([v['radius'] for v in self.vessels])
        
        if self.flow_rates:
            total_flow = sum(self.flow_rates)
        else:
            total_flow = 0
        
        return {
            'total_vessels': len(self.vessels),
            'arterial_vessels': arterial_count,
            'venous_vessels': venous_count,
            'total_length': total_length,
            'average_radius': avg_radius,
            'total_flow': total_flow,
            'max_depth': max(v['depth'] for v in self.vessels) if self.vessels else 0
        }
