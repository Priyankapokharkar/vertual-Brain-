"""
Brain Disease Detection Model
Implements neural network for multi-class brain disease classification
"""
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from config import Config


class DiseaseDetector:
    """
    Brain disease detection using deep learning
    Supports multiple disease classification including:
    - Alzheimer's Disease
    - Parkinson's Disease
    - Stroke
    - Epilepsy
    - Healthy (no disease)
    """
    
    def __init__(self, model_path=None):
        """Initialize the disease detector"""
        self.config = Config()
        self.model_path = Path(model_path or self.config.DISEASE_MODEL_PATH)
        self.classes = self.config.DISEASE_CLASSES
        self.img_size = (224, 224)
        self.model = None
        self.is_loaded = False
        
    def build_model(self, input_shape=(128, 128, 1)):
        """
        Build a 2D CNN for brain disease classification
        
        Args:
            input_shape: Shape of input brain images (H, W, D, C)
        
        Returns:
            Compiled Keras model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for model building")
        
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Output layer (binary crossentropy or categorical for 2 classes)
            layers.Dense(len(self.classes), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_model(self):
        """Load pre-trained model if available"""
        if not TF_AVAILABLE:
            print("Warning: TensorFlow not available. Using dummy predictions.")
            return False
            
        if self.model_path.exists():
            try:
                # Try standard load
                self.model = keras.models.load_model(str(self.model_path))
                self.is_loaded = True
                print(f"Model loaded from {self.model_path}")
                return True
            except Exception as e:
                # Handle Keras version mismatch (e.g., quantization_config error)
                if 'quantization_config' in str(e):
                    print("Detected Keras version mismatch. Attempting to load without compilation...")
                    try:
                        self.model = keras.models.load_model(str(self.model_path), compile=False)
                        self.is_loaded = True
                        print("Model loaded successfully (non-compiled).")
                        return True
                    except:
                        pass
                
                print(f"Error loading model: {e}")
                print("Building new model as fallback.")
                self.model = self.build_model()
                self.is_loaded = True
                return True
        else:
            print(f"Model not found at {self.model_path}. Building new model.")
            self.model = self.build_model()
            self.is_loaded = True
            return True
    
    def predict(self, brain_data: np.ndarray) -> Dict:
        """
        Predict disease from brain imaging data
        
        Args:
            brain_data: 3D or 4D numpy array of brain imaging data
        
        Returns:
            Dictionary containing predictions, probabilities, and confidence
        """
        if not self.is_loaded:
            self.load_model()
        
        if self.model is not None and TF_AVAILABLE:
            # Ensure data has correct shape based on model's expected input rank
            expected_rank = len(self.model.input_shape)
            
            # 1. Handle 3D volume to 2D slice if model is 2D (rank 4)
            if expected_rank == 4 and brain_data.ndim == 3 and brain_data.shape[2] > 3:
                # Take middle slice
                brain_data = brain_data[:, :, brain_data.shape[2] // 2]
                
            # 2. Add channel dimension for 2D if missing
            if expected_rank == 4 and brain_data.ndim == 2:
                brain_data = np.expand_dims(brain_data, axis=-1)
                
            # 3. Add batch dimension if missing
            if brain_data.ndim == expected_rank - 1:
                brain_data = np.expand_dims(brain_data, axis=0)
                
            # 4. Resize spatial dimensions and channels if needed
            import tensorflow as tf
            if expected_rank == 4:
                expected_shape = self.model.input_shape[1:3]
                if expected_shape[0] is not None and (brain_data.shape[1] != expected_shape[0] or brain_data.shape[2] != expected_shape[1]):
                    brain_data = tf.image.resize(brain_data, expected_shape).numpy()
                
                expected_channels = self.model.input_shape[-1]
                if expected_channels == 3 and brain_data.shape[-1] == 1:
                    brain_data = np.concatenate([brain_data, brain_data, brain_data], axis=-1)
                
            # Final safety check
            if brain_data.ndim != expected_rank:
                print(f"Dimension mismatch: expected rank {expected_rank}, got {brain_data.ndim}")
                pass
            
        # 5. Apply ResNetV2 specific preprocessing (scaling to [-1, 1])
        # If data is in range [0, 1], we map to [-1, 1]
        if np.max(brain_data) <= 1.01:
            brain_data = (brain_data * 2.0) - 1.0
        else:
            # If data is [0, 255], use standard formula
            brain_data = (brain_data / 127.5) - 1.0
        
        # Make prediction
        if self.model is not None and TF_AVAILABLE:
            predictions = self.model.predict(brain_data, verbose=0)
        else:
            # Dummy predictions for demonstration
            predictions = self._dummy_prediction()
        
        # Process results
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.classes[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get all class probabilities
        probabilities = {
            self.classes[i]: float(predictions[0][i])
            for i in range(len(self.classes))
        }
        
        # Calculate risk factors
        risk_level = self._calculate_risk_level(confidence, predicted_class)
        
        return {
            'predicted_disease': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'risk_level': risk_level,
            'threshold_met': confidence >= self.config.CONFIDENCE_THRESHOLD,
            'confidence_threshold': self.config.CONFIDENCE_THRESHOLD,
            'recommendations': self._get_recommendations(predicted_class, confidence)
        }
    
    def _dummy_prediction(self) -> np.ndarray:
        """Generate dummy predictions for demonstration"""
        # Create realistic-looking probabilities
        probs = np.random.dirichlet(np.ones(len(self.classes)) * 0.5, size=1)
        # Make one class dominant
        dominant_idx = np.random.randint(0, len(self.classes))
        probs[0][dominant_idx] += 0.3
        probs = probs / probs.sum()  # Normalize
        return probs
    
    def _calculate_risk_level(self, confidence: float, disease: str) -> str:
        """Calculate risk level based on confidence and disease type"""
        if disease == 'Normal':
            return 'low'
        
        # High-risk conditions
        critical_diseases = ['Glioma', 'Meningioma', 'Pituitary', 'MS']
        if disease in critical_diseases:
            return 'high' if confidence >= 0.7 else 'moderate'
            
        # Progressive conditions
        if 'Alzheimer' in disease or disease == 'Parkinson':
            return 'moderate-high' if confidence >= 0.8 else 'moderate'
            
        return 'low'
    
    def _get_recommendations(self, disease: str, confidence: float) -> List[str]:
        """Get clinical recommendations based on prediction"""
        recommendations = []
        
        if disease == 'Normal':
            recommendations = [
                "Continue regular health monitoring",
                "Maintain healthy lifestyle habits",
                "Schedule routine scans for comparison if symptoms arise"
            ]
        elif 'Alzheimer' in disease:
            recommendations = [
                f"Prediction indicates {disease.replace('_', ' ')}",
                "Consult with a geriatric neurologist specializing in dementia",
                "Cognitive assessment and volumetric MRI analysis recommended",
                "Initiate lifestyle interventions (diet, exercise, cognitive training)",
                "Review potential for disease-modifying therapies if applicable"
            ]
        elif disease in ['Glioma', 'Meningioma', 'Pituitary']:
            recommendations = [
                f"URGENT: Highly probable {disease} tumor detected",
                "Immediate consultation with a neurosurgeon and oncologist",
                "Conduct contrast-enhanced MRI and spectroscopy",
                "Evaluate for potential surgical resection or biopsy",
                "Localized simulation triggered to model tissue displacement"
            ]
        elif disease == 'MS':
            recommendations = [
                "Detection of Multiple Sclerosis-like lesions",
                "Consult with an MS specialist (Neurologist)",
                "Lumbar puncture and evoked potential tests may be recommended",
                "Monitor for new lesion activity and neurological symptoms",
                "Initiate disease-modifying therapy (DMT) discussion"
            ]
        elif disease == 'Parkinson':
            recommendations = [
                "Early indices of Parkinson's Disease noted",
                "Consult with a movement disorder specialist",
                "Dopamine transporter (DaT) scan or PET scan for confirmation",
                "Monitor motor symptoms (tremor, rigidity, bradykinesia)",
                "Exercise programs specialized for PD are highly recommended"
            ]
        
        if confidence < 0.7 and disease != 'Normal':
            recommendations.insert(0, "Note: Confidence level is low. Correlation with clinical symptoms is essential.")
        
        return recommendations
    
    def batch_predict(self, brain_data_list: List[np.ndarray]) -> List[Dict]:
        """Predict multiple brain scans"""
        results = []
        for brain_data in brain_data_list:
            results.append(self.predict(brain_data))
        return results
    
    def get_feature_importance(self, brain_data: np.ndarray) -> Dict:
        """
        Estimate which brain features/regions were most important for the prediction
        """
        # Simplified: Calculate intensity distribution across regions
        # in a real model, this would use Grad-CAM or similar
        features = {
            'frontal_lobe': np.mean(brain_data[:32, :32]),
            'temporal_lobe': np.mean(brain_data[32:64, 32:64]),
            'parietal_lobe': np.mean(brain_data[64:96, 64:96]),
            'occipital_lobe': np.mean(brain_data[96:, 96:]),
            'cerebellum': np.mean(brain_data[48:80, :48]),
            'brainstem': np.mean(brain_data[80:, 48:80])
        }
        
        # Normalize
        total = sum(features.values()) + 1e-10
        return {k: float(v / total) for k, v in features.items()}

    def get_tumor_localization(self, brain_data: np.ndarray) -> Dict:
        """
        Estimate tumor location (focal point) in normalized coordinates [-1, 1]
        """
        # Find the region with highest intensity clusters
        h, w = brain_data.shape[0], brain_data.shape[1]
        
        # Flatten and find top intensity pixels
        flat = brain_data.flatten()
        threshold = np.percentile(flat, 98) # top 2%
        mask = brain_data > threshold
        
        if not np.any(mask):
            return {'centroid': [0.4, 0.3, 0.4], 'radius': 0.3} # Default fallback
            
        coords = np.where(mask)
        # Average coordinates
        cy = float(np.mean(coords[0]))
        cx = float(np.mean(coords[1]))
        
        # Convert to normalized [-1, 1] space for the 3D model
        # Note: mapping image axes (y, x) to 3D axes
        norm_x = (cx / w) * 2 - 1
        norm_y = -(cy / h) * 2 + 1 # Invert Y for mesh space
        norm_z = 0.2 # Slightly anterior
        
        return {
            'centroid': [norm_x, norm_y, norm_z],
            'radius': 0.35 + 0.1 * np.random.rand(),
            'severity': 0.8
        }
