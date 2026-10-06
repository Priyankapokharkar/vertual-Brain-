"""
Neural Mass Model implementation
Simulates population-level neural dynamics
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.integrate import odeint
from scipy.signal import periodogram


class NeuralMassModel:
    """
    Neural mass model for simulating brain region activity
    Implements Jansen-Rit model for EEG-like signals
    """
    
    def __init__(self, num_regions: int = 90, dt: float = 0.001):
        """
        Initialize neural mass model
        
        Args:
            num_regions: Number of brain regions
            dt: Time step for simulation (seconds)
        """
        self.num_regions = num_regions
        self.dt = dt
        
        # Model parameters (Jansen-Rit default values)
        self.params = {
            'A': 3.25,      # Maximum amplitude of EPSP (mV)
            'B': 22.0,      # Maximum amplitude of IPSP (mV)
            'a': 100.0,     # Inverse of time constant for excitatory (1/s)
            'b': 50.0,      # Inverse of time constant for inhibitory (1/s)
            'v0': 6.0,      # Firing threshold (mV)
            'e0': 2.5,      # Half of maximum firing rate (spikes/s)
            'r': 0.56,      # Steepness of sigmoid
            'C': 135.0,     # Connectivity constant
            'C1': 1.0,      # Pyramidal to excitatory interneuron
            'C2': 0.8,      # Excitatory interneuron to pyramidal
            'C3': 0.25,     # Inhibitory interneuron to pyramidal
            'C4': 0.25,     # Pyramidal to inhibitory interneuron
        }
        
        # State variables for each region
        self.states = None
        self.reset_states()
        
    def reset_states(self):
        """Reset all state variables"""
        # Each region has 6 state variables
        # [pyramidal potential, excitatory potential, inhibitory potential,
        #  pyramidal derivative, excitatory derivative, inhibitory derivative]
        self.states = np.random.randn(self.num_regions, 6) * 0.1
    
    def sigmoid(self, v: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 2 * self.params['e0'] / (1 + np.exp(self.params['r'] * (self.params['v0'] - v)))
    
    def jansen_rit_dynamics(self, state: np.ndarray, t: float, 
                           input_signal: float, coupling: np.ndarray) -> np.ndarray:
        """
        Jansen-Rit model differential equations
        
        Args:
            state: Current state [6 variables]
            t: Time
            input_signal: External input
            coupling: Coupling from other regions
        
        Returns:
            Derivatives of state variables
        """
        A, B, a, b = self.params['A'], self.params['B'], self.params['a'], self.params['b']
        C, C1, C2, C3, C4 = [self.params[k] for k in ['C', 'C1', 'C2', 'C3', 'C4']]
        
        # Unpack state variables
        y0, y1, y2, y3, y4, y5 = state
        
        # Input to pyramidal population
        py_input = C * input_signal + C2 * self.sigmoid(C1 * y0) - C3 * self.sigmoid(C4 * y0) + coupling
        
        # Differential equations
        dy0 = y3
        dy1 = y4
        dy2 = y5
        dy3 = A * a * self.sigmoid(y1 - y2) - 2 * a * y3 - a**2 * y0
        dy4 = A * a * py_input - 2 * a * y4 - a**2 * y1
        dy5 = B * b * C4 * self.sigmoid(C3 * y0) - 2 * b * y5 - b**2 * y2
        
        return np.array([dy0, dy1, dy2, dy3, dy4, dy5])
    
    def simulate_single_region(self, duration: float, 
                              input_function=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate single isolated region
        
        Args:
            duration: Simulation duration (seconds)
            input_function: External input function(t)
        
        Returns:
            (time_array, activity_array)
        """
        if input_function is None:
            input_function = lambda t: 220 + 20 * np.sin(2 * np.pi * 10 * t)  # 10 Hz oscillation
        
        t = np.arange(0, duration, self.dt)
        
        # Initial state
        state0 = np.random.randn(6) * 0.1
        
        # Simulate
        states = []
        state = state0
        
        for time in t:
            # RK4 integration
            k1 = self.jansen_rit_dynamics(state, time, input_function(time), 0)
            k2 = self.jansen_rit_dynamics(state + 0.5 * self.dt * k1, time + 0.5 * self.dt, input_function(time), 0)
            k3 = self.jansen_rit_dynamics(state + 0.5 * self.dt * k2, time + 0.5 * self.dt, input_function(time), 0)
            k4 = self.jansen_rit_dynamics(state + self.dt * k3, time + self.dt, input_function(time), 0)
            
            state = state + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            states.append(state.copy())
        
        states = np.array(states)
        activity = states[:, 1] - states[:, 2]  # Excitatory - Inhibitory (EEG-like)
        
        return t, activity
    
    def simulate_network(self, duration: float, 
                        connectivity_matrix: np.ndarray,
                        coupling_strength: float = 10.0,
                        noise_level: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate coupled network of brain regions
        
        Args:
            duration: Simulation duration (seconds)
            connectivity_matrix: Connectivity between regions
            coupling_strength: Global coupling strength
            noise_level: Noise amplitude
        
        Returns:
            (time_array, activity_matrix) where activity_matrix is (time x regions)
        """
        t = np.arange(0, duration, self.dt)
        n_steps = len(t)
        
        # Initialize activity matrix
        activity = np.zeros((n_steps, self.num_regions))
        
        # Normalize connectivity
        conn_norm = connectivity_matrix / (connectivity_matrix.sum(axis=1, keepdims=True) + 1e-10)
        
        # Simulate each time step
        for i, time in enumerate(t):
            # Calculate coupling for each region
            if i > 0:
                # Coupling from other regions
                coupling = coupling_strength * (conn_norm @ activity[i-1, :])
            else:
                coupling = np.zeros(self.num_regions)
            
            # Update each region
            for r in range(self.num_regions):
                # External input with noise
                external_input = 220 + noise_level * np.random.randn()
                
                # RK4 step
                k1 = self.jansen_rit_dynamics(self.states[r], time, external_input, coupling[r])
                k2 = self.jansen_rit_dynamics(self.states[r] + 0.5 * self.dt * k1, 
                                             time + 0.5 * self.dt, external_input, coupling[r])
                k3 = self.jansen_rit_dynamics(self.states[r] + 0.5 * self.dt * k2, 
                                             time + 0.5 * self.dt, external_input, coupling[r])
                k4 = self.jansen_rit_dynamics(self.states[r] + self.dt * k3, 
                                             time + self.dt, external_input, coupling[r])
                
                self.states[r] = self.states[r] + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
                
                # Record activity (EEG-like: excitatory - inhibitory)
                activity[i, r] = self.states[r, 1] - self.states[r, 2]
        
        return t, activity
    
    def calculate_functional_connectivity(self, activity: np.ndarray) -> np.ndarray:
        """
        Calculate functional connectivity from simulated activity
        
        Args:
            activity: Activity matrix (time x regions)
        
        Returns:
            Functional connectivity matrix (regions x regions)
        """
        # Pearson correlation
        fc = np.corrcoef(activity.T)
        return fc
    
    def analyze_oscillations(self, time: np.ndarray, activity: np.ndarray) -> Dict:
        """
        Analyze oscillatory properties of neural activity
        
        Args:
            time: Time array
            activity: Activity time series
        
        Returns:
            Dictionary of oscillation metrics
        """
        # Power spectral density
        fs = 1 / self.dt
        freqs, psd = periodogram(activity, fs=fs)
        
        # Find dominant frequency
        dominant_idx = np.argmax(psd)
        dominant_freq = freqs[dominant_idx]
        
        # Band power
        def band_power(f_low, f_high):
            mask = (freqs >= f_low) & (freqs <= f_high)
            return np.trapz(psd[mask], freqs[mask])
        
        delta = band_power(1, 4)
        theta = band_power(4, 8)
        alpha = band_power(8, 13)
        beta = band_power(13, 30)
        gamma = band_power(30, 100)
        
        return {
            'dominant_frequency': float(dominant_freq),
            'dominant_power': float(psd[dominant_idx]),
            'band_powers': {
                'delta': float(delta),
                'theta': float(theta),
                'alpha': float(alpha),
                'beta': float(beta),
                'gamma': float(gamma)
            }
        }
    
    def simulate_disease_condition(self, duration: float, 
                                  connectivity_matrix: np.ndarray,
                                  disease_type: str = 'healthy') -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate neural activity under different disease conditions
        
        Args:
            duration: Simulation duration
            connectivity_matrix: Structural connectivity
            disease_type: Type of disease to simulate
        
        Returns:
            (time_array, activity_matrix)
        """
        # Modify parameters based on disease
        original_params = self.params.copy()
        coupling_strength = 10.0
        
        if disease_type == 'alzheimer':
            # Reduced connectivity and altered excitability
            connectivity_matrix = connectivity_matrix * 0.6
            self.params['C'] = 100.0
            coupling_strength = 5.0
            
        elif disease_type == 'parkinson':
            # Altered oscillations (increased beta)
            self.params['a'] = 120.0
            coupling_strength = 12.0
            
        elif disease_type == 'epilepsy':
            # Hyperexcitability
            self.params['C'] = 180.0
            coupling_strength = 20.0
            
        elif disease_type == 'tumor':
            # Focal hyperexcitability and mass effect
            # We simulate this by increasing excitability in specific regions
            # and increasing global coupling to simulate pressure/spread
            self.params['C'] = 160.0
            self.params['a'] = 110.0  # Faster excitatory time constant
            coupling_strength = 15.0
            # Increase connectivity in the "tumor area" (random Focal point for now)
            tumor_center = np.random.randint(0, self.num_regions)
            affected_regions = [tumor_center] + list(np.random.choice(self.num_regions, 5, replace=False))
            connectivity_matrix[affected_regions, :] *= 1.5
            
        elif disease_type == 'stroke':
            # Lesion simulation: remove random regions
            num_lesioned = int(0.1 * self.num_regions)
            lesion_idx = np.random.choice(self.num_regions, num_lesioned, replace=False)
            connectivity_matrix[lesion_idx, :] = 0
            connectivity_matrix[:, lesion_idx] = 0
            coupling_strength = 10.0
            coupling_strength = 10.0
        
        # Simulate
        t, activity = self.simulate_network(duration, connectivity_matrix, coupling_strength)
        
        # Restore original parameters
        self.params = original_params
        
        return t, activity
