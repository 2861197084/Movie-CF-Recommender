"""
Automatic MovieLens dataset downloader
Following academic research standards for reproducible data acquisition
"""

import os
import requests
import zipfile
import pandas as pd
from typing import Optional, Tuple
from urllib.parse import urlparse
import hashlib
import shutil

from utils.logger import get_logger
from config import cfg

logger = get_logger("DatasetDownloader")

class MovieLensDownloader:
    """
    Automatic downloader for MovieLens datasets with verification

    Supports multiple MovieLens dataset versions:
    - ml-latest-small: Small dataset for development/testing (~100k ratings)
    - ml-latest: Full dataset (~27M ratings)
    - ml-25m: Stable 25M ratings dataset
    - ml-100k: Classic 100k ratings dataset
    """

    DATASET_URLS = {
        "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "ml-latest": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
        "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    }

    # Expected file structures for each dataset
    DATASET_FILES = {
        "ml-latest-small": {
            "ratings": "ratings.csv",
            "movies": "movies.csv",
            "links": "links.csv",
            "tags": "tags.csv"
        },
        "ml-latest": {
            "ratings": "ratings.csv",
            "movies": "movies.csv",
            "links": "links.csv",
            "tags": "tags.csv"
        },
        "ml-25m": {
            "ratings": "ratings.csv",
            "movies": "movies.csv",
            "links": "links.csv",
            "tags": "tags.csv",
            "genome-scores": "genome-scores.csv",
            "genome-tags": "genome-tags.csv"
        },
        "ml-100k": {
            "ratings": "u.data",
            "movies": "u.item",
            "users": "u.user"
        }
    }

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize downloader

        Args:
            data_dir: Directory to store downloaded data (defaults to config)
        """
        self.data_dir = data_dir or cfg.data.dataset_path
        os.makedirs(self.data_dir, exist_ok=True)

    def download_dataset(self, dataset_name: str = "ml-latest-small",
                        force_redownload: bool = False) -> Tuple[str, str]:
        """
        Download and extract MovieLens dataset

        Args:
            dataset_name: Name of dataset to download
            force_redownload: Whether to redownload if already exists

        Returns:
            Tuple of (ratings_file_path, movies_file_path)
        """
        logger.log_phase(f"Downloading MovieLens Dataset: {dataset_name}")

        if dataset_name not in self.DATASET_URLS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.DATASET_URLS.keys())}")

        # Check if dataset already exists
        if not force_redownload and self._dataset_exists(dataset_name):
            logger.info(f"Dataset {dataset_name} already exists, skipping download")
            return self._get_dataset_paths(dataset_name)

        # Download dataset
        url = self.DATASET_URLS[dataset_name]
        zip_path = self._download_file(url)

        # Extract dataset
        extract_path = self._extract_dataset(zip_path, dataset_name)

        # Convert to standard format if needed
        ratings_path, movies_path = self._standardize_format(extract_path, dataset_name)

        # Cleanup zip file
        os.remove(zip_path)

        logger.info(f"Dataset {dataset_name} successfully downloaded and prepared")
        return ratings_path, movies_path

    def _dataset_exists(self, dataset_name: str) -> bool:
        """Check if dataset files already exist"""
        try:
            ratings_path, movies_path = self._get_dataset_paths(dataset_name)
            return os.path.exists(ratings_path) and os.path.exists(movies_path)
        except:
            return False

    def _get_dataset_paths(self, dataset_name: str) -> Tuple[str, str]:
        """Get expected paths for dataset files"""
        ratings_path = os.path.join(self.data_dir, "ratings.csv")
        movies_path = os.path.join(self.data_dir, "movies.csv")
        return ratings_path, movies_path

    def _download_file(self, url: str) -> str:
        """Download file from URL with progress tracking"""
        filename = os.path.basename(urlparse(url).path)
        filepath = os.path.join(self.data_dir, filename)

        logger.info(f"Downloading {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        if downloaded_size % (1024*1024) == 0:  # Log every MB
                            logger.info(f"Download progress: {progress:.1f}% ({downloaded_size/1024/1024:.1f}MB)")

        logger.info(f"Download completed: {filepath}")
        return filepath

    def _extract_dataset(self, zip_path: str, dataset_name: str) -> str:
        """Extract dataset from zip file"""
        extract_dir = os.path.join(self.data_dir, "temp_extract")

        logger.info(f"Extracting {zip_path}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find the actual dataset directory (usually named after dataset)
        extracted_contents = os.listdir(extract_dir)
        dataset_dir = None

        for item in extracted_contents:
            item_path = os.path.join(extract_dir, item)
            if os.path.isdir(item_path) and dataset_name in item:
                dataset_dir = item_path
                break

        if not dataset_dir:
            # If no matching directory, use the first directory found
            for item in extracted_contents:
                item_path = os.path.join(extract_dir, item)
                if os.path.isdir(item_path):
                    dataset_dir = item_path
                    break

        if not dataset_dir:
            raise ValueError(f"Could not find dataset directory in extracted files")

        logger.info(f"Dataset extracted to: {dataset_dir}")
        return dataset_dir

    def _standardize_format(self, extract_path: str, dataset_name: str) -> Tuple[str, str]:
        """Convert dataset to standardized format"""
        logger.info("Converting dataset to standard format")

        dataset_files = self.DATASET_FILES[dataset_name]

        if dataset_name == "ml-100k":
            # Special handling for ml-100k format
            return self._convert_ml100k(extract_path)
        else:
            # Standard CSV format datasets
            return self._copy_standard_files(extract_path, dataset_files)

    def _convert_ml100k(self, extract_path: str) -> Tuple[str, str]:
        """Convert ML-100k format to standard CSV format"""
        logger.info("Converting ML-100k format to standard CSV")

        # Convert ratings (u.data format: user_id item_id rating timestamp)
        ratings_input = os.path.join(extract_path, "u.data")
        ratings_output = os.path.join(self.data_dir, "ratings.csv")

        if os.path.exists(ratings_input):
            ratings_df = pd.read_csv(ratings_input, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
            ratings_df.to_csv(ratings_output, index=False)
            logger.info(f"Converted ratings: {ratings_output}")

        # Convert movies (u.item format is more complex, we'll create a simplified version)
        movies_input = os.path.join(extract_path, "u.item")
        movies_output = os.path.join(self.data_dir, "movies.csv")

        if os.path.exists(movies_input):
            # u.item format: movie_id | movie_title | release_date | video_release_date | IMDb_URL | unknown | Action | Adventure | ...
            movies_data = []
            with open(movies_input, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        movie_id = int(parts[0])
                        title = parts[1]
                        # Extract genre information from binary indicators (columns 5-23)
                        genre_names = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                                     'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                     'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
                        genres = []
                        if len(parts) >= 24:
                            for i, genre in enumerate(genre_names):
                                if i + 5 < len(parts) and parts[i + 5] == '1':
                                    genres.append(genre)
                        genre_str = '|'.join(genres) if genres else 'unknown'
                        movies_data.append([movie_id, title, genre_str])

            movies_df = pd.DataFrame(movies_data, columns=['movieId', 'title', 'genres'])
            movies_df.to_csv(movies_output, index=False)
            logger.info(f"Converted movies: {movies_output}")

        # Cleanup temp directory
        shutil.rmtree(os.path.dirname(extract_path))

        return ratings_output, movies_output

    def _copy_standard_files(self, extract_path: str, dataset_files: dict) -> Tuple[str, str]:
        """Copy standard format files to data directory"""
        ratings_source = os.path.join(extract_path, dataset_files["ratings"])
        movies_source = os.path.join(extract_path, dataset_files["movies"])

        ratings_dest = os.path.join(self.data_dir, "ratings.csv")
        movies_dest = os.path.join(self.data_dir, "movies.csv")

        if not os.path.exists(ratings_source):
            raise FileNotFoundError(f"Ratings file not found: {ratings_source}")
        if not os.path.exists(movies_source):
            raise FileNotFoundError(f"Movies file not found: {movies_source}")

        shutil.copy2(ratings_source, ratings_dest)
        shutil.copy2(movies_source, movies_dest)

        logger.info(f"Copied ratings to: {ratings_dest}")
        logger.info(f"Copied movies to: {movies_dest}")

        # Cleanup temp directory
        shutil.rmtree(os.path.dirname(extract_path))

        return ratings_dest, movies_dest

    def get_dataset_info(self, dataset_name: str = "ml-latest-small") -> dict:
        """Get information about a dataset"""
        if dataset_name not in self.DATASET_URLS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        info = {
            "name": dataset_name,
            "url": self.DATASET_URLS[dataset_name],
            "files": self.DATASET_FILES[dataset_name],
            "description": self._get_dataset_description(dataset_name)
        }

        return info

    def _get_dataset_description(self, dataset_name: str) -> str:
        """Get description for dataset"""
        descriptions = {
            "ml-latest-small": "Small dataset with ~100,000 ratings for development and testing",
            "ml-latest": "Latest full dataset with ~27 million ratings",
            "ml-25m": "Stable dataset with 25 million ratings",
            "ml-100k": "Classic dataset with 100,000 ratings (historical format)"
        }
        return descriptions.get(dataset_name, "MovieLens dataset")

    def list_available_datasets(self) -> list:
        """List all available datasets"""
        datasets = []
        for name in self.DATASET_URLS.keys():
            info = self.get_dataset_info(name)
            datasets.append({
                "name": name,
                "description": info["description"],
                "url": info["url"]
            })
        return datasets

def download_movielens_data(dataset_name: str = "ml-latest-small",
                          data_dir: Optional[str] = None,
                          force_redownload: bool = False) -> Tuple[str, str]:
    """
    Convenience function to download MovieLens data

    Args:
        dataset_name: Dataset to download
        data_dir: Directory to store data
        force_redownload: Whether to redownload if exists

    Returns:
        Tuple of (ratings_file_path, movies_file_path)
    """
    downloader = MovieLensDownloader(data_dir)
    return downloader.download_dataset(dataset_name, force_redownload)