"""
Vessel Blockage Detection
Identifies and analyzes vascular stenosis and occlusions
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter1d


class BlockageDetector:
    """
    Detects and analyzes vessel blockages
    """
    
    def __init__(self, sensitivity: float = 0.7):
        """
        Initialize blockage detector
        
        Args:
            sensitivity: Detection sensitivity (0-1)
        """
        self.sensitivity = sensitivity
        self.blockages = []
        
    def analyze_vessel(self, vessel: Dict, flow_data: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze single vessel for blockages
        
        Args:
            vessel: Vessel dictionary with geometry
            flow_data: Optional flow rate data along vessel
        
        Returns:
            Analysis results
        """
        result = {
            'vessel_id': vessel['id'],
            'has_blockage': False,
            'blockage_locations': [],
            'severity': 0.0,
            'risk_level': 'low'
        }
        
        # Check for explicit blockage
        if 'blockage_severity' in vessel and vessel['blockage_severity'] > 0:
            result['has_blockage'] = True
            result['severity'] = vessel['blockage_severity']
            result['blockage_locations'].append({
                'position': vessel['start'],
                'severity': vessel['blockage_severity'],
                'type': 'stenosis' if vessel['blockage_severity'] < 0.9 else 'occlusion'
            })
        
        # Analyze flow if available
        if flow_data is not None:
            flow_blockages = self._detect_flow_anomalies(flow_data)
            if flow_blockages:
                result['has_blockage'] = True
                result['blockage_locations'].extend(flow_blockages)
                result['severity'] = max(result['severity'], 
                                        max(b['severity'] for b in flow_blockages))
        
        # Calculate risk level
        result['risk_level'] = self._calculate_risk_level(result['severity'])
        
        return result
    
    def _detect_flow_anomalies(self, flow_data: np.ndarray) -> List[Dict]:
        """
        Detect anomalies in flow data
        
        Args:
            flow_data: 1D array of flow rates along vessel
        
        Returns:
            List of detected blockages
        """
        anomalies = []
        
        # Smooth flow data
        smoothed = gaussian_filter1d(flow_data, sigma=2)
        
        # Calculate gradient (sudden drops indicate blockage)
        gradient = np.gradient(smoothed)
        
        # Detect significant drops
        threshold = -self.sensitivity * np.std(gradient)
        
        drop_indices = np.where(gradient < threshold)[0]
        
        for idx in drop_indices:
            # Calculate severity based on drop magnitude
            drop_magnitude = abs(gradient[idx])
            severity = min(1.0, drop_magnitude / (np.max(np.abs(gradient)) + 1e-10))
            
            if severity > 0.3:  # Only report significant blockages
                anomalies.append({
                    'position_index': int(idx),
                    'severity': float(severity),
                    'type': 'flow_anomaly',
                    'flow_reduction': float(drop_magnitude)
                })
        
        return anomalies
    
    def _calculate_risk_level(self, severity: float) -> str:
        """
        Calculate clinical risk level
        
        Args:
            severity: Blockage severity (0-1)
        
        Returns:
            Risk level string
        """
        if severity >= 0.8:
            return 'critical'
        elif severity >= 0.6:
            return 'high'
        elif severity >= 0.3:
            return 'moderate'
        else:
            return 'low'
    
    def analyze_network(self, vessels: List[Dict], flow_rates: List[float]) -> Dict:
        """
        Analyze entire vessel network
        
        Args:
            vessels: List of vessel dictionaries
            flow_rates: List of flow rates for each vessel
        
        Returns:
            Network analysis results
        """
        self.blockages = []
        
        for i, vessel in enumerate(vessels):
            flow = flow_rates[i] if i < len(flow_rates) else None
            analysis = self.analyze_vessel(vessel, flow_data=None)  # Simplified
            
            if analysis['has_blockage']:
                self.blockages.append({
                    'vessel_id': vessel['id'],
                    'vessel_type': vessel['type'],
                    'severity': analysis['severity'],
                    'risk_level': analysis['risk_level'],
                    'location': vessel['start']
                })
        
        # Calculate overall statistics
        num_blockages = len(self.blockages)
        
        if num_blockages > 0:
            severities = [b['severity'] for b in self.blockages]
            avg_severity = np.mean(severities)
            max_severity = np.max(severities)
            
            critical_count = sum(1 for b in self.blockages if b['risk_level'] == 'critical')
            high_count = sum(1 for b in self.blockages if b['risk_level'] == 'high')
        else:
            avg_severity = 0.0
            max_severity = 0.0
            critical_count = 0
            high_count = 0
        
        return {
            'total_vessels_analyzed': len(vessels),
            'blockages_detected': num_blockages,
            'blockage_rate': num_blockages / len(vessels) if vessels else 0,
            'average_severity': avg_severity,
            'max_severity': max_severity,
            'critical_blockages': critical_count,
            'high_risk_blockages': high_count,
            'blockages': self.blockages
        }
    
    def simulate_intervention(self, vessel_id: int, vessels: List[Dict]) -> Dict:
        """
        Simulate intervention (stenting, angioplasty) outcome
        
        Args:
            vessel_id: ID of vessel to treat
            vessels: List of vessels
        
        Returns:
            Predicted outcome
        """
        vessel = vessels[vessel_id]
        
        before_severity = vessel.get('blockage_severity', 0)
        
        # Simulate intervention success
        success_rate = 0.85  # 85% success rate
        
        if np.random.rand() < success_rate:
            after_severity = before_severity * 0.1  # 90% reduction
            outcome = 'successful'
        else:
            after_severity = before_severity * 0.5  # 50% reduction
            outcome = 'partial'
        
        flow_improvement = (1 - after_severity) / (1 - before_severity + 1e-10) if before_severity < 1 else 2.0
        
        return {
            'outcome': outcome,
            'severity_before': before_severity,
            'severity_after': after_severity,
            'flow_improvement': flow_improvement,
            'recommendations': self._get_intervention_recommendations(after_severity)
        }
    
    def _get_intervention_recommendations(self, post_severity: float) -> List[str]:
        """Get recommendations based on post-intervention severity"""
        recommendations = []
        
        if post_severity < 0.2:
            recommendations.append("Excellent result - routine follow-up in 6 months")
        elif post_severity < 0.4:
            recommendations.append("Good result - monitor with imaging in 3 months")
            recommendations.append("Continue antiplatelet therapy")
        else:
            recommendations.append("Suboptimal result - consider repeat intervention")
            recommendations.append("Close monitoring required")
            recommendations.append("Optimize medical management")
        
        return recommendations
    
    def generate_heatmap(self, vessels: List[Dict], resolution: int = 50) -> np.ndarray:
        """
        Generate 3D heatmap of blockage risk
        
        Args:
            vessels: List of vessels
            resolution: Grid resolution
        
        Returns:
            3D heatmap array
        """
        # Create 3D grid
        heatmap = np.zeros((resolution, resolution, resolution))
        
        # Get spatial bounds
        all_points = []
        for vessel in vessels:
            all_points.extend([vessel['start'], vessel['end']])
        all_points = np.array(all_points)
        
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        
        # Map vessels to grid
        for vessel in vessels:
            severity = vessel.get('blockage_severity', 0)
            
            if severity > 0:
                start = np.array(vessel['start'])
                
                # Convert to grid coordinates
                grid_pos = ((start - min_coords) / (max_coords - min_coords + 1e-10) * (resolution - 1)).astype(int)
                grid_pos = np.clip(grid_pos, 0, resolution - 1)
                
                # Add to heatmap
                heatmap[grid_pos[0], grid_pos[1], grid_pos[2]] += severity
        
        # Smooth heatmap
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=2)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def export_report(self) -> Dict:
        """Export comprehensive blockage detection report"""
        if not self.blockages:
            return {
                'status': 'No blockages detected',
                'recommendations': ['Continue regular monitoring', 'Maintain healthy lifestyle']
            }
        
        # Sort blockages by severity
        sorted_blockages = sorted(self.blockages, key=lambda x: x['severity'], reverse=True)
        
        # Generate recommendations
        recommendations = []
        
        critical_count = sum(1 for b in self.blockages if b['risk_level'] == 'critical')
        
        if critical_count > 0:
            recommendations.append(f"URGENT: {critical_count} critical blockage(s) detected")
            recommendations.append("Immediate clinical evaluation required")
            recommendations.append("Consider emergency intervention")
        
        high_risk_count = sum(1 for b in self.blockages if b['risk_level'] == 'high')
        if high_risk_count > 0:
            recommendations.append(f"{high_risk_count} high-risk blockage(s) require attention")
            recommendations.append("Schedule vascular assessment")
        
        return {
            'summary': {
                'total_blockages': len(self.blockages),
                'critical': critical_count,
                'high_risk': high_risk_count,
                'average_severity': np.mean([b['severity'] for b in self.blockages])
            },
            'blockages': sorted_blockages[:10],  # Top 10 most severe
            'recommendations': recommendations,
            'timestamp': 'Generated by Virtual Brain Inference'
        }
