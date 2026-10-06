"""
Configuration file for Virtual Brain Inference application
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'vbi-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # Upload settings
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    ALLOWED_EXTENSIONS = {'nii', 'nii.gz', 'dcm', 'png', 'jpg', 'jpeg', 'mat', 'json'}
    
    # Model settings
    MODELS_DIR = BASE_DIR / 'models' / 'pretrained'
    DISEASE_MODEL_PATH = MODELS_DIR / 'multi_disease_detector.h5'
    
    # Data directories
    DATA_DIR = BASE_DIR / 'data'
    STATIC_DIR = BASE_DIR / 'static'
    TEMPLATES_DIR = BASE_DIR / 'templates'
    
    # Connectome settings
    CONNECTOME_PATH = DATA_DIR / 'sample_connectome.json'
    BRAIN_REGIONS_PATH = DATA_DIR / 'brain_regions.json'
    VESSEL_NETWORK_PATH = DATA_DIR / 'vessel_network.json'
    
    # 3D Model paths
    BRAIN_MODEL_PATH = STATIC_DIR / 'models' / 'brain.obj'
    
    # Simulation parameters
    SIMULATION_TIMESTEP = 0.001  # 1ms
    SIMULATION_DURATION = 10.0  # 10 seconds
    NUM_BRAIN_REGIONS = 90  # AAL atlas default
    
    # Disease detection settings
    DISEASE_CLASSES = [
        'Alzheimer_Mild', 'Alzheimer_Moderate', 'Alzheimer_VeryMild', 
        'Glioma', 'MS', 'Meningioma', 'Normal', 'Parkinson', 'Pituitary'
    ]
    CONFIDENCE_THRESHOLD = 0.75
    
    # Visualization settings
    BLOCKAGE_SEVERITY_LEVELS = {
        'none': 0.0,
        'mild': 0.3,
        'moderate': 0.6,
        'severe': 0.85
    }
    
    # API settings
    API_VERSION = 'v1'
    RESULTS_PER_PAGE = 20
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = BASE_DIR / 'logs' / 'vbi.log'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env='default'):
    """Get configuration based on environment"""
    return config.get(env, config['default'])
