"""
Utility functions for the prediction module.

This module provides helper functions for common tasks such as:
- Image information retrieval
- Result saving (JSON, CSV)
- Input validation
- File operations

Author: Member 4 - Prediction & Inference Team
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple
from PIL import Image
from .config import SUPPORTED_IMAGE_FORMATS, MIN_IMAGE_SIZE, MAX_IMAGE_SIZE

# Setup logger
logger = logging.getLogger(__name__)


def validate_image_path(image_path: str) -> bool:
    """
    Validate image file path and format.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        bool: True if valid
        
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If format not supported
        
    Example:
        >>> validate_image_path("test.jpg")
        True
    """
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Check if it's a file
    if not os.path.isfile(image_path):
        raise ValueError(f"Path is not a file: {image_path}")
    
    # Check file extension
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            f"Unsupported image format: {file_ext}\n"
            f"Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
        )
    
    logger.debug(f"Validated image: {image_path}")
    return True


def validate_image_size(image_path: str) -> Tuple[int, int]:
    """
    Validate image size is within acceptable range.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        Tuple[int, int]: Image size (width, height)
        
    Raises:
        ValueError: If image size is too small or too large
    """
    img = Image.open(image_path)
    width, height = img.size
    
    # Check minimum size
    if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
        raise ValueError(
            f"Image too small: {width}x{height}. "
            f"Minimum size: {MIN_IMAGE_SIZE[0]}x{MIN_IMAGE_SIZE[1]}"
        )
    
    # Check maximum size
    if width > MAX_IMAGE_SIZE[0] or height > MAX_IMAGE_SIZE[1]:
        raise ValueError(
            f"Image too large: {width}x{height}. "
            f"Maximum size: {MAX_IMAGE_SIZE[0]}x{MAX_IMAGE_SIZE[1]}"
        )
    
    return (width, height)


def get_image_info(image_path: str) -> Dict[str, Any]:
    """
    Get detailed information about an image.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        Dict[str, Any]: Image information including size, mode, format, etc.
        
    Example:
        >>> info = get_image_info("test.jpg")
        >>> print(info['size'])
        (1024, 768)
    """
    img = Image.open(image_path)
    
    return {
        'filename': os.path.basename(image_path),
        'full_path': os.path.abspath(image_path),
        'size': img.size,  # (width, height)
        'width': img.size[0],
        'height': img.size[1],
        'mode': img.mode,  # 'RGB', 'L', etc.
        'format': img.format,  # 'JPEG', 'PNG', etc.
        'file_size_bytes': os.path.getsize(image_path)
    }


def save_predictions_to_json(
    predictions: List[Dict[str, Any]],
    output_path: str,
    indent: int = 2
) -> str:
    """
    Save prediction results to JSON file.
    
    Args:
        predictions (List[Dict]): List of prediction dictionaries
        output_path (str): Output file path
        indent (int): JSON indentation level
        
    Returns:
        str: Path to saved file
        
    Example:
        >>> save_predictions_to_json(results, "predictions.json")
        'predictions.json'
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=indent, ensure_ascii=False)
        
        logger.info(f"Saved predictions to JSON: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        raise


def save_predictions_to_csv(
    predictions: List[Dict[str, Any]],
    output_path: str
) -> str:
    """
    Save prediction results to CSV file.
    
    Args:
        predictions (List[Dict]): List of prediction dictionaries
        output_path (str): Output file path
        
    Returns:
        str: Path to saved file
        
    Example:
        >>> save_predictions_to_csv(results, "predictions.csv")
        'predictions.csv'
        
    Reference:
        Pandas to_csv: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
    """
    try:
        import pandas as pd
        
        # Flatten predictions for CSV (exclude nested top_k_predictions)
        flat_predictions = []
        for pred in predictions:
            flat_pred = {
                'image_path': pred['image_path'],
                'predicted_class': pred['predicted_class'],
                'confidence': pred['confidence']
            }
            flat_predictions.append(flat_pred)
        
        df = pd.DataFrame(flat_predictions)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved predictions to CSV: {output_path}")
        return output_path
        
    except ImportError:
        logger.error("pandas is required for CSV export. Install with: pip install pandas")
        raise
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        raise


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes (int): File size in bytes
        
    Returns:
        str: Formatted size (e.g., "1.5 MB")
        
    Example:
        >>> format_file_size(1536000)
        '1.46 MB'
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def ensure_dir_exists(directory: str) -> str:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory (str): Directory path
        
    Returns:
        str: Directory path
        
    Example:
        >>> ensure_dir_exists("results/predictions")
        'results/predictions'
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    return directory


# Module-level test code
if __name__ == "__main__":
    print("Utility Module Test")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - validate_image_path()")
    print("  - validate_image_size()")
    print("  - get_image_info()")
    print("  - save_predictions_to_json()")
    print("  - save_predictions_to_csv()")
    print("  - format_file_size()")
    print("  - ensure_dir_exists()")
