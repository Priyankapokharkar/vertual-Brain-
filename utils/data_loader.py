"""
Utility functions for data loading and processing
"""
import numpy as np
from pathlib import Path
from typing import Union, Optional

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


def load_brain_data(file_path: Union[str, Path], 
                    data_type: str = 'auto') -> np.ndarray:
    """
    Load brain imaging data from various formats
    
    Args:
        file_path: Path to data file
        data_type: Type of data ('nifti', 'dicom', 'image', 'auto')
    
    Returns:
        Loaded brain data as numpy array
    """
    file_path = Path(file_path)
    
    if data_type == 'auto':
        # Auto-detect format
        ext = file_path.suffix.lower()
        if ext in ['.nii', '.gz']:
            data_type = 'nifti'
        elif ext in ['.dcm']:
            data_type = 'dicom'
        elif ext in ['.png', '.jpg', '.jpeg']:
            data_type = 'image'
        else:
            raise ValueError(f"Cannot auto-detect format for {ext}")
    
    if data_type == 'nifti':
        return load_nifti(file_path)
    elif data_type == 'dicom':
        return load_dicom(file_path)
    elif data_type == 'image':
        return load_image(file_path)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def load_dicom(file_path: Path) -> np.ndarray:
    """Load DICOM format file"""
    import pydicom
    dicom = pydicom.dcmread(str(file_path))
    return dicom.pixel_array


def load_nifti(file_path: Path) -> np.ndarray:
    """Load NIfTI format file"""
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is required for NIfTI files")
    
    img = nib.load(str(file_path))
    return img.get_fdata()


def load_image(file_path: Path) -> np.ndarray:
    """Load standard image format"""
    from PIL import Image
    img = Image.open(file_path).convert('L')
    return np.array(img)


def save_brain_data(data: np.ndarray, 
                   file_path: Union[str, Path],
                   format: str = 'nifti') -> None:
    """
    Save brain data to file
    
    Args:
        data: Brain data array
        file_path: Output file path
        format: Output format
    """
    file_path = Path(file_path)
    
    if format == 'nifti':
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel required for NIfTI output")
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, str(file_path))
    elif format == 'numpy':
        np.save(file_path, data)
    else:
        raise ValueError(f"Unsupported format: {format}")


def validate_brain_data(data: np.ndarray, 
                       min_dims: int = 3,
                       max_dims: int = 4) -> bool:
    """
    Validate brain imaging data
    
    Args:
        data: Brain data array
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array")
    
    if data.ndim < min_dims or data.ndim > max_dims:
        raise ValueError(f"Data must have {min_dims}-{max_dims} dimensions, got {data.ndim}")
    
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")
    
    if np.any(np.isinf(data)):
        raise ValueError("Data contains infinite values")
    
    return True
