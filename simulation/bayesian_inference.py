"""
Bayesian Inference for personalized brain models
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal


class BayesianInference:
    """
    Bayesian parameter inference for Virtual Brain models
    Estimates optimal parameters given observed brain data
    """
    
    def __init__(self, num_params: int = 5):
        """
        Initialize Bayesian inference engine
        
        Args:
            num_params: Number of parameters to infer
        """
        self.num_params = num_params
        self.prior_means = None
        self.prior_stds = None
        self.posterior_samples = None
        
    def set_priors(self, means: np.ndarray, stds: np.ndarray):
        """
        Set prior distributions for parameters
        
        Args:
            means: Prior means for each parameter
            stds: Prior standard deviations
        """
        self.prior_means = np.array(means)
        self.prior_stds = np.array(stds)
    
    def log_prior(self, params: np.ndarray) -> float:
        """
        Calculate log prior probability
        
        Args:
            params: Parameter vector
        
        Returns:
            Log prior probability
        """
        if self.prior_means is None:
            # Uniform prior
            return 0.0
        
        # Gaussian prior
        log_p = -0.5 * np.sum(((params - self.prior_means) / self.prior_stds) ** 2)
        log_p -= np.sum(np.log(self.prior_stds))
        log_p -= 0.5 * self.num_params * np.log(2 * np.pi)
        
        return log_p
    
    def log_likelihood(self, params: np.ndarray, 
                      observed_data: np.ndarray,
                      simulator: Callable) -> float:
        """
        Calculate log likelihood of parameters given data
        
        Args:
            params: Parameter vector
            observed_data: Observed brain activity or connectivity
            simulator: Function that generates data from parameters
        
        Returns:
            Log likelihood
        """
        # Simulate data with given parameters
        simulated_data = simulator(params)
        
        # Calculate likelihood (assuming Gaussian noise)
        residuals = observed_data - simulated_data
        sigma = 0.1  # Noise standard deviation
        
        log_l = -0.5 * np.sum((residuals / sigma) ** 2)
        log_l -= len(residuals) * np.log(sigma)
        log_l -= 0.5 * len(residuals) * np.log(2 * np.pi)
        
        return log_l
    
    def log_posterior(self, params: np.ndarray,
                     observed_data: np.ndarray,
                     simulator: Callable) -> float:
        """
        Calculate log posterior probability
        
        Args:
            params: Parameter vector
            observed_data: Observed data
            simulator: Simulation function
        
        Returns:
            Log posterior probability
        """
        log_p = self.log_prior(params)
        
        # Check if parameters are valid
        if not np.isfinite(log_p) or np.any(params < 0):
            return -np.inf
        
        log_l = self.log_likelihood(params, observed_data, simulator)
        
        return log_p + log_l
    
    def maximum_a_posteriori(self, observed_data: np.ndarray,
                           simulator: Callable,
                           initial_params: Optional[np.ndarray] = None) -> Dict:
        """
        Find Maximum A Posteriori (MAP) estimate
        
        Args:
            observed_data: Observed brain data
            simulator: Simulation function
            initial_params: Initial parameter guess
        
        Returns:
            Dictionary with MAP estimate and optimization results
        """
        if initial_params is None:
            if self.prior_means is not None:
                initial_params = self.prior_means
            else:
                initial_params = np.ones(self.num_params)
        
        # Define objective (negative log posterior)
        def objective(params):
            return -self.log_posterior(params, observed_data, simulator)
        
        # Optimize
        result = minimize(objective, initial_params, 
                         method='L-BFGS-B',
                         bounds=[(0, None)] * self.num_params)
        
        return {
            'map_estimate': result.x,
            'log_posterior': -result.fun,
            'success': result.success,
            'message': result.message,
            'num_iterations': result.nit
        }
    
    def metropolis_hastings(self, observed_data: np.ndarray,
                           simulator: Callable,
                           n_samples: int = 1000,
                           burn_in: int = 200,
                           proposal_std: float = 0.1) -> Dict:
        """
        Metropolis-Hastings MCMC sampling
        
        Args:
            observed_data: Observed data
            simulator: Simulation function
            n_samples: Number of samples to draw
            burn_in: Number of burn-in samples
            proposal_std: Standard deviation for proposal distribution
        
        Returns:
            Dictionary with samples and acceptance rate
        """
        # Initialize
        if self.prior_means is not None:
            current_params = self.prior_means.copy()
        else:
            current_params = np.ones(self.num_params)
        
        current_log_post = self.log_posterior(current_params, observed_data, simulator)
        
        samples = []
        accepted = 0
        
        # MCMC loop
        for i in range(n_samples + burn_in):
            # Propose new parameters
            proposal = current_params + np.random.randn(self.num_params) * proposal_std
            
            # Ensure positive parameters
            if np.any(proposal < 0):
                continue
            
            # Calculate posterior
            proposal_log_post = self.log_posterior(proposal, observed_data, simulator)
            
            # Accept/reject
            log_alpha = proposal_log_post - current_log_post
            
            if np.log(np.random.rand()) < log_alpha:
                current_params = proposal
                current_log_post = proposal_log_post
                accepted += 1
            
            # Store sample (after burn-in)
            if i >= burn_in:
                samples.append(current_params.copy())
        
        self.posterior_samples = np.array(samples)
        
        return {
            'samples': self.posterior_samples,
            'mean': np.mean(self.posterior_samples, axis=0),
            'std': np.std(self.posterior_samples, axis=0),
            'acceptance_rate': accepted / (n_samples + burn_in)
        }
    
    def estimate_functional_connectivity_params(self, 
                                               observed_fc: np.ndarray,
                                               structural_conn: np.ndarray) -> Dict:
        """
        Estimate parameters that best explain observed functional connectivity
        
        Args:
            observed_fc: Observed functional connectivity matrix
            structural_conn: Structural connectivity matrix
        
        Returns:
            Estimated parameters
        """
        # Parameters: [coupling_strength, noise_level, excitability, inhibition, time_constant]
        self.num_params = 5
        
        # Set reasonable priors
        prior_means = np.array([10.0, 20.0, 3.0, 20.0, 0.01])
        prior_stds = np.array([5.0, 10.0, 1.0, 5.0, 0.005])
        self.set_priors(prior_means, prior_stds)
        
        # Simple simulator for FC
        def simulate_fc(params):
            coupling, noise, exc, inh, tau = params
            
            # Simplified FC simulation based on structural connectivity
            # In practice, this would run full neural mass simulation
            fc_sim = structural_conn.copy()
            fc_sim = fc_sim * coupling / 10.0
            fc_sim = fc_sim + np.random.randn(*fc_sim.shape) * noise / 100.0
            
            # Ensure symmetric
            fc_sim = (fc_sim + fc_sim.T) / 2
            
            # Flatten upper triangle for comparison
            triu_idx = np.triu_indices_from(fc_sim, k=1)
            return fc_sim[triu_idx]
        
        # Extract upper triangle of observed FC
        triu_idx = np.triu_indices_from(observed_fc, k=1)
        observed_fc_vec = observed_fc[triu_idx]
        
        # Find MAP estimate
        result = self.maximum_a_posteriori(observed_fc_vec, simulate_fc)
        
        return {
            'coupling_strength': result['map_estimate'][0],
            'noise_level': result['map_estimate'][1],
            'excitability': result['map_estimate'][2],
            'inhibition': result['map_estimate'][3],
            'time_constant': result['map_estimate'][4],
            'optimization_info': result
        }
    
    def model_comparison(self, observed_data: np.ndarray,
                        simulators: List[Callable],
                        model_names: List[str]) -> Dict:
        """
        Compare different models using Bayesian model selection
        
        Args:
            observed_data: Observed data
            simulators: List of simulation functions for each model
            model_names: Names of models
        
        Returns:
            Model comparison results
        """
        log_evidences = []
        
        for simulator in simulators:
            # Use Laplace approximation for evidence
            map_result = self.maximum_a_posteriori(observed_data, simulator)
            
            # Approximate log evidence (simplified)
            log_evidence = map_result['log_posterior']
            log_evidences.append(log_evidence)
        
        # Calculate model probabilities
        log_evidences = np.array(log_evidences)
        log_evidences = log_evidences - np.max(log_evidences)  # Numerical stability
        evidences = np.exp(log_evidences)
        model_probs = evidences / evidences.sum()
        
        results = {
            'model_names': model_names,
            'log_evidences': log_evidences.tolist(),
            'model_probabilities': model_probs.tolist(),
            'best_model': model_names[np.argmax(model_probs)]
        }
        
        return results
    
    def predictive_distribution(self, test_input: np.ndarray,
                               simulator: Callable) -> Dict:
        """
        Generate predictive distribution using posterior samples
        
        Args:
            test_input: Input for prediction
            simulator: Simulation function
        
        Returns:
            Predictive statistics
        """
        if self.posterior_samples is None:
            raise ValueError("Must run MCMC sampling first")
        
        predictions = []
        
        for params in self.posterior_samples:
            pred = simulator(params)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        return {
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'median': np.median(predictions, axis=0),
            'percentile_2.5': np.percentile(predictions, 2.5, axis=0),
            'percentile_97.5': np.percentile(predictions, 97.5, axis=0)
        }
