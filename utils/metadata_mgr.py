"""
Metadata manager module for handling image metadata and dataset splits.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split


class MetadataManager:
    """
    Manages metadata generation and dataset splitting.
    """
    
    def __init__(self, metadata_dir: Path):
        """
        Initialize the metadata manager.
        
        Args:
            metadata_dir: Directory to save metadata files
        """
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def save_image_metadata(
        self, 
        image_id: str, 
        original_path: Path,
        processing_metadata: dict
    ) -> bool:
        """
        Save metadata for a processed image as JSON.
        
        Args:
            image_id: Unique identifier for the image
            original_path: Path to the original image file
            processing_metadata: Dictionary containing processing information
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            metadata = {
                'image_id': image_id,
                'original_path': str(original_path),
                'original_filename': original_path.name,
                'source_dataset': self._get_dataset_name(original_path),
                'original_height': processing_metadata['original_size'][0],
                'original_width': processing_metadata['original_size'][1],
                'processed_height': processing_metadata['processed_size'][0],
                'processed_width': processing_metadata['processed_size'][1],
                'i_channel_min': processing_metadata['i_channel_min'],
                'i_channel_max': processing_metadata['i_channel_max'],
                'i_channel_normalized': processing_metadata['i_channel_normalized']
            }
            
            output_path = self.metadata_dir / f"{image_id}.json"
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving metadata for {image_id}: {str(e)}")
            return False
    
    def _get_dataset_name(self, image_path: Path) -> str:
        """
        Determine the dataset name from the image path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dataset name (e.g., 'div2k', 'bossbase', 'unknown')
        """
        path_str = str(image_path).lower()
        if 'div2k' in path_str:
            return 'div2k'
        elif 'bossbase' in path_str or 'boss' in path_str:
            return 'bossbase'
        else:
            return 'unknown'
    
    def save_watermark_metadata(
        self,
        watermark_id: str,
        original_path: Path,
        scrambling_metadata: dict
    ) -> bool:
        """
        Save metadata for a scrambled watermark as JSON.
        
        Args:
            watermark_id: Unique identifier for the watermark
            original_path: Path to the original watermark file
            scrambling_metadata: Dictionary containing scrambling information
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            metadata = {
                'watermark_id': watermark_id,
                'original_path': str(original_path),
                'original_filename': original_path.name,
                'scrambling_algorithm': scrambling_metadata['algorithm'],
                'acm_iterations': scrambling_metadata['iterations'],
                'original_watermark_dimensions': scrambling_metadata['original_dimensions'],
                'scrambled_watermark_size': scrambling_metadata['scrambled_size'],
                'scrambled_path': scrambling_metadata['scrambled_path']
            }
            
            output_path = self.metadata_dir / f"watermark_{watermark_id}.json"
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logging.info(f"Saved watermark metadata: {output_path.name}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving watermark metadata for {watermark_id}: {str(e)}")
            return False


def create_splits(
    all_ids: List[str], 
    splits_dir: Path,
    ratios: List[float] = [0.7, 0.15, 0.15],
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/validation/test splits and save to text files.
    
    Uses sklearn.model_selection.train_test_split with a fixed random seed
    for reproducibility.
    
    Args:
        all_ids: List of all image IDs
        splits_dir: Directory to save split files
        ratios: List of [train, validation, test] ratios (must sum to 1.0)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    # Validate ratios
    if not abs(sum(ratios) - 1.0) < 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
    
    train_ratio, val_ratio, test_ratio = ratios
    
    # First split: separate train from (val + test)
    train_ids, temp_ids = train_test_split(
        all_ids,
        test_size=(val_ratio + test_ratio),
        random_state=random_seed
    )
    
    # Second split: separate val from test
    # Adjust the ratio for the second split
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1 - val_size_adjusted),
        random_state=random_seed
    )
    
    # Create splits directory
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits to text files
    _save_split_file(splits_dir / "train.txt", train_ids)
    _save_split_file(splits_dir / "val.txt", val_ids)
    _save_split_file(splits_dir / "test.txt", test_ids)
    
    logging.info(f"Created splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    
    return train_ids, val_ids, test_ids


def _save_split_file(file_path: Path, ids: List[str]) -> None:
    """
    Save a list of IDs to a text file (one per line).
    
    Args:
        file_path: Path to the output file
        ids: List of image IDs
    """
    with open(file_path, 'w') as f:
        for img_id in sorted(ids):
            f.write(f"{img_id}\n")
    
    logging.info(f"Saved {len(ids)} IDs to {file_path}")
