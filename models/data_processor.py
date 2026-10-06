"""
Data preprocessing utilities for brain imaging data
"""
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import json

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


class DataProcessor:
    """
    Handles preprocessing of various brain imaging formats
    Supports: NIfTI (.nii, .nii.gz), DICOM, PNG/JPG
    """
    
    def __init__(self, target_shape=(128, 128, 128)):
        """
        Initialize data processor
        
        Args:
            target_shape: Target shape for resizing images
        """
        self.target_shape = target_shape
        
    def load_nifti(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load NIfTI format brain imaging data
        
        Args:
            file_path: Path to .nii or .nii.gz file
        
        Returns:
            3D numpy array of brain data
        """
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required to load NIfTI files")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load NIfTI file
        nii_img = nib.load(str(file_path))
        data = nii_img.get_fdata()
        
        return data
    
    def load_image(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load 2D image (PNG, JPG) and convert to 3D volume
        
        Args:
            file_path: Path to image file
        
        Returns:
            3D numpy array (simulated volume)
        """
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required to load images")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load image
        img = Image.open(file_path).convert('L')  # Convert to grayscale
        img_array = np.array(img)
        
        # Simulate 3D volume by stacking
        depth = self.target_shape[2]
        volume = np.stack([img_array] * depth, axis=-1)
        
        return volume
    
    def load_dicom(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load DICOM format brain imaging data
        
        Args:
            file_path: Path to .dcm file or directory containing .dcm files
            
        Returns:
            3D numpy array of brain data
        """
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required to load DICOM files")
            
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File/Directory not found: {file_path}")
            
        if file_path.is_file():
            # Single DICOM file
            dicom = pydicom.dcmread(str(file_path))
            pixel_array = dicom.pixel_array
            
            # If it's a 2D slice, simulate 3D volume
            if len(pixel_array.shape) == 2:
                depth = self.target_shape[2]
                return np.stack([pixel_array] * depth, axis=-1)
            return pixel_array
            
        elif file_path.is_dir():
            # Directory of DICOM slices
            slices = []
            for dcm_file in sorted(file_path.glob('*.dcm')):
                dicom = pydicom.dcmread(str(dcm_file))
                slices.append(dicom.pixel_array)
                
            if not slices:
                raise ValueError(f"No .dcm files found in directory {file_path}")
                
            volume = np.stack(slices, axis=-1)
            return volume
            
        raise ValueError("Unsupported path type for DICOM")
    
    def normalize(self, data: np.ndarray, method='minmax') -> np.ndarray:
        """
        Normalize imaging data
        
        Args:
            data: Input data array
            method: Normalization method ('minmax', 'zscore', 'percentile')
        
        Returns:
            Normalized data
        """
        if method == 'minmax':
            # Min-max normalization to [0, 1]
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max - data_min > 0:
                normalized = (data - data_min) / (data_max - data_min)
            else:
                normalized = data
                
        elif method == 'zscore':
            # Z-score normalization
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                normalized = (data - mean) / std
            else:
                normalized = data
                
        elif method == 'percentile':
            # Percentile-based normalization
            p1, p99 = np.percentile(data, [1, 99])
            normalized = np.clip((data - p1) / (p99 - p1), 0, 1)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def resize(self, data: np.ndarray, target_shape: Optional[Tuple] = None) -> np.ndarray:
        """
        Resize 3D volume to target shape
        
        Args:
            data: Input 3D array
            target_shape: Target shape (H, W, D)
        
        Returns:
            Resized data
        """
        if target_shape is None:
            target_shape = self.target_shape
        
        # Use simple interpolation (trilinear)
        from scipy.ndimage import zoom
        
        # If data is 4D (e.g. fMRI), take the first timepoint
        if len(data.shape) == 4:
            data = data[0]
            
        current_shape = data.shape
        zoom_factors = [
            target_shape[i] / current_shape[i]
            for i in range(3)
        ]
        
        resized = zoom(data, zoom_factors, order=1)
        
        return resized
    
    def skull_strip(self, data: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Simple skull stripping using thresholding
        
        Args:
            data: Input brain volume
            threshold: Intensity threshold for masking
        
        Returns:
            Skull-stripped data
        """
        # Create binary mask
        mask = data > (threshold * np.max(data))
        
        # Apply morphological operations (simplified)
        from scipy.ndimage import binary_erosion, binary_dilation
        
        mask = binary_erosion(mask, iterations=2)
        mask = binary_dilation(mask, iterations=3)
        
        # Apply mask
        stripped = data * mask
        
        return stripped
    
    def extract_features(self, data: np.ndarray) -> dict:
        """
        Extract statistical features from brain volume
        
        Args:
            data: Input brain volume
        
        Returns:
            Dictionary of features
        """
        features = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'percentile_25': float(np.percentile(data, 25)),
            'percentile_75': float(np.percentile(data, 75)),
            'skewness': float(self._skewness(data)),
            'kurtosis': float(self._kurtosis(data)),
            'entropy': float(self._entropy(data))
        }
        
        return features
    
    def preprocess(self, file_path: Union[str, Path], 
                   normalize: bool = True,
                   skull_strip: bool = False,
                   resize: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            file_path: Path to brain imaging file
            normalize: Whether to normalize data
            skull_strip: Whether to perform skull stripping
            resize: Whether to resize to target shape
        
        Returns:
            Preprocessed brain volume
        """
        file_path = Path(file_path)
        
        # Load data based on file extension
        if file_path.is_dir() or file_path.suffix.lower() == '.dcm':
            # Assume directory contains DICOM slices or it's a direct .dcm file
            data = self.load_dicom(file_path)
        elif file_path.suffix in ['.nii', '.gz']:
            data = self.load_nifti(file_path)
        elif file_path.suffix in ['.png', '.jpg', '.jpeg']:
            data = self.load_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Apply preprocessing steps
        if skull_strip:
            data = self.skull_strip(data)
        
        if resize:
            data = self.resize(data)
        
        if normalize:
            data = self.normalize(data, method='minmax')
        
        return data
    
    def augment(self, data: np.ndarray) -> np.ndarray:
        """
        Data augmentation for training
        
        Args:
            data: Input brain volume
        
        Returns:
            Augmented data
        """
        # Random rotation
        if np.random.rand() > 0.5:
            axes = np.random.choice([0, 1, 2], size=2, replace=False)
            k = np.random.randint(1, 4)
            data = np.rot90(data, k=k, axes=axes)
        
        # Random flip
        if np.random.rand() > 0.5:
            axis = np.random.randint(0, 3)
            data = np.flip(data, axis=axis)
        
        # Random noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, data.shape)
            data = data + noise
            data = np.clip(data, 0, 1)
        
        # Random brightness
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            data = data * factor
            data = np.clip(data, 0, 1)
        
        return data
    
    def create_synthetic_data(self, shape: Optional[Tuple] = None) -> np.ndarray:
        """
        Create synthetic brain data for testing
        
        Args:
            shape: Shape of synthetic volume
        
        Returns:
            Synthetic brain volume
        """
        if shape is None:
            shape = self.target_shape
        
        # Create base volume with anatomical-like structure
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, shape[0]),
            np.linspace(-1, 1, shape[1]),
            np.linspace(-1, 1, shape[2]),
            indexing='ij'
        )
        
        # Create brain-like ellipsoid
        brain = np.exp(-(x**2 + y**2 + 1.5*z**2) / 0.5)
        
        # Add some structure variations
        structure = np.sin(5 * x) * np.cos(5 * y) * np.sin(3 * z)
        brain = brain + 0.1 * structure
        
        # Add noise
        noise = np.random.normal(0, 0.05, shape)
        brain = brain + noise
        
        # Normalize
        brain = self.normalize(brain, method='minmax')
        
        return brain
    
    @staticmethod
    def _skewness(data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod
    def _kurtosis(data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    @staticmethod
    def _entropy(data: np.ndarray, bins: int = 256) -> float:
        """Calculate entropy"""
        hist, _ = np.histogram(data, bins=bins)
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
