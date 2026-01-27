"""
Dataset downloader module for automatically downloading and extracting datasets.
"""
import logging
from pathlib import Path
from typing import Optional
import zipfile
import requests
from tqdm import tqdm


class DatasetDownloader:
    """
    Handles automatic downloading and extraction of datasets.
    """
    
    DIV2K_TRAIN_HR_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    
    def __init__(self, data_root: Path):
        """
        Initialize the dataset downloader.
        
        Args:
            data_root: Root directory for storing raw datasets
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, destination: Path, chunk_size: int = 8192) -> bool:
        """
        Download a file from URL with progress bar.
        
        Args:
            url: URL to download from
            destination: Path where the file should be saved
            chunk_size: Size of chunks to download at a time (bytes)
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            logging.info(f"Downloading from {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logging.info(f"Successfully downloaded to {destination}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to download {url}: {str(e)}")
            if destination.exists():
                destination.unlink()
            return False
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """
        Extract a ZIP archive.
        
        Args:
            zip_path: Path to the ZIP file
            extract_to: Directory to extract contents to
            
        Returns:
            True if extraction was successful, False otherwise
        """
        try:
            logging.info(f"Extracting {zip_path.name}")
            extract_to.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                for member in tqdm(members, desc="Extracting"):
                    zip_ref.extract(member, extract_to)
            
            logging.info(f"Successfully extracted to {extract_to}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to extract {zip_path}: {str(e)}")
            return False
    
    def download_div2k(self, force_redownload: bool = False) -> Optional[Path]:
        """
        Check if DIV2K dataset exists (manual download required).
        
        Args:
            force_redownload: Ignored (kept for compatibility)
            
        Returns:
            Path to the dataset directory, or None if not found
        """
        div2k_dir = self.data_root / "div2k"
        
        # Check if directory exists and has PNG files
        if div2k_dir.exists():
            png_files = list(div2k_dir.glob("*.png"))
            if len(png_files) > 0:
                logging.info(f"DIV2K dataset found at {div2k_dir} ({len(png_files)} images)")
                return div2k_dir
        
        logging.error(f"DIV2K dataset not found at {div2k_dir}")
        logging.error("Please manually download DIV2K_train_HR.zip and extract images to data/raw/div2k/")
        return None
