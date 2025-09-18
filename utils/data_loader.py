"""
Data loading and preprocessing utilities for MovieLens dataset
Following academic research standards for reproducibility
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import os

from config import cfg
from utils.logger import get_logger
from utils.dataset_downloader import download_movielens_data

logger = get_logger("DataLoader")

class MovieLensLoader:
    """
    MovieLens dataset loader with academic-standard preprocessing
    """

    def __init__(self, config=None):
        self.config = config or cfg
        self.ratings_df = None
        self.movies_df = None
        self.user_item_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.inverse_user_mapping = None
        self.inverse_item_mapping = None

    def load_data(self, dataset_name: str = "ml-latest-small",
                  auto_download: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load MovieLens ratings and movies data with automatic download

        Args:
            dataset_name: Name of MovieLens dataset to use
            auto_download: Whether to automatically download if not found

        Returns:
            Tuple of (ratings_df, movies_df)
        """
        logger.log_phase("Data Loading")

        # Construct file paths
        ratings_path = os.path.join(self.config.data.dataset_path, self.config.data.ratings_file)
        movies_path = os.path.join(self.config.data.dataset_path, self.config.data.movies_file)

        # Check if files exist, download if not
        if not (os.path.exists(ratings_path) and os.path.exists(movies_path)):
            if auto_download:
                logger.info(f"Dataset not found, automatically downloading {dataset_name}...")
                try:
                    ratings_path, movies_path = download_movielens_data(
                        dataset_name=dataset_name,
                        data_dir=self.config.data.dataset_path,
                        force_redownload=False
                    )
                    logger.info("Dataset download completed successfully")
                except Exception as e:
                    logger.error(f"Failed to download dataset: {e}")
                    raise FileNotFoundError(
                        f"Dataset files not found and automatic download failed. "
                        f"Please manually download MovieLens dataset to {self.config.data.dataset_path}"
                    )
            else:
                raise FileNotFoundError(
                    f"Dataset files not found: {ratings_path}, {movies_path}. "
                    f"Set auto_download=True or manually download MovieLens dataset."
                )

        # Load data
        logger.info(f"Loading ratings from: {ratings_path}")
        self.ratings_df = pd.read_csv(ratings_path)

        logger.info(f"Loading movies from: {movies_path}")
        self.movies_df = pd.read_csv(movies_path)

        # Log basic statistics
        logger.info(f"Loaded {len(self.ratings_df)} ratings")
        logger.info(f"Loaded {len(self.movies_df)} movies")
        logger.info(f"Number of unique users: {self.ratings_df['userId'].nunique()}")
        logger.info(f"Number of unique movies: {self.ratings_df['movieId'].nunique()}")

        return self.ratings_df, self.movies_df

    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the ratings data with filtering and cleaning

        Returns:
            Preprocessed ratings DataFrame
        """
        logger.log_phase("Data Preprocessing")

        if self.ratings_df is None:
            raise ValueError("Data must be loaded first. Call load_data().")

        # Create a copy to avoid modifying original data
        processed_ratings = self.ratings_df.copy()

        # Log initial statistics
        initial_users = processed_ratings['userId'].nunique()
        initial_items = processed_ratings['movieId'].nunique()
        initial_ratings = len(processed_ratings)

        logger.info(f"Initial statistics:")
        logger.info(f"  Users: {initial_users}")
        logger.info(f"  Items: {initial_items}")
        logger.info(f"  Ratings: {initial_ratings}")

        # Filter users with minimum ratings
        if self.config.data.min_ratings_per_user > 1:
            user_counts = processed_ratings['userId'].value_counts()
            valid_users = user_counts[user_counts >= self.config.data.min_ratings_per_user].index
            processed_ratings = processed_ratings[processed_ratings['userId'].isin(valid_users)]

            logger.info(f"Filtered users with < {self.config.data.min_ratings_per_user} ratings")
            logger.info(f"Remaining users: {processed_ratings['userId'].nunique()}")

        # Filter items with minimum ratings
        if self.config.data.min_ratings_per_item > 1:
            item_counts = processed_ratings['movieId'].value_counts()
            valid_items = item_counts[item_counts >= self.config.data.min_ratings_per_item].index
            processed_ratings = processed_ratings[processed_ratings['movieId'].isin(valid_items)]

            logger.info(f"Filtered items with < {self.config.data.min_ratings_per_item} ratings")
            logger.info(f"Remaining items: {processed_ratings['movieId'].nunique()}")

        # Log final statistics
        final_users = processed_ratings['userId'].nunique()
        final_items = processed_ratings['movieId'].nunique()
        final_ratings = len(processed_ratings)

        logger.info(f"Final statistics after preprocessing:")
        logger.info(f"  Users: {final_users} ({final_users/initial_users*100:.1f}%)")
        logger.info(f"  Items: {final_items} ({final_items/initial_items*100:.1f}%)")
        logger.info(f"  Ratings: {final_ratings} ({final_ratings/initial_ratings*100:.1f}%)")

        # Calculate sparsity
        sparsity = 1 - (final_ratings / (final_users * final_items))
        logger.info(f"  Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")

        self.ratings_df = processed_ratings
        return processed_ratings

    def create_user_item_matrix(self) -> csr_matrix:
        """
        Create user-item interaction matrix

        Returns:
            Sparse user-item matrix
        """
        logger.log_phase("Creating User-Item Matrix")

        if self.ratings_df is None:
            raise ValueError("Data must be preprocessed first.")

        # Create user and item mappings
        unique_users = sorted(self.ratings_df['userId'].unique())
        unique_items = sorted(self.ratings_df['movieId'].unique())

        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.inverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.inverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}

        # Map to matrix indices
        user_indices = self.ratings_df['userId'].map(self.user_mapping)
        item_indices = self.ratings_df['movieId'].map(self.item_mapping)
        ratings = self.ratings_df['rating'].values

        # Create sparse matrix
        n_users = len(unique_users)
        n_items = len(unique_items)

        self.user_item_matrix = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )

        logger.info(f"Created user-item matrix: {n_users} users × {n_items} items")
        logger.info(f"Matrix density: {self.user_item_matrix.nnz / (n_users * n_items):.6f}")

        return self.user_item_matrix

    def train_test_split(self, random_state=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets

        Returns:
            Tuple of (train_df, test_df)
        """
        logger.log_phase("Train-Test Split")

        if self.ratings_df is None:
            raise ValueError("Data must be preprocessed first.")

        random_state = random_state or self.config.experiment.random_seed

        train_df, test_df = train_test_split(
            self.ratings_df,
            test_size=self.config.data.test_ratio,
            random_state=random_state,
            stratify=None  # Could stratify by user or rating if needed
        )

        logger.info(f"Train set: {len(train_df)} ratings ({len(train_df)/len(self.ratings_df)*100:.1f}%)")
        logger.info(f"Test set: {len(test_df)} ratings ({len(test_df)/len(self.ratings_df)*100:.1f}%)")

        # Verify no data leakage (users in both sets)
        train_users = set(train_df['userId'])
        test_users = set(test_df['userId'])
        overlap_users = train_users.intersection(test_users)

        logger.info(f"Users in both train/test: {len(overlap_users)} "
                   f"({len(overlap_users)/len(train_users.union(test_users))*100:.1f}%)")

        return train_df, test_df

    def get_dataset_statistics(self) -> Dict:
        """
        Get comprehensive dataset statistics for academic reporting

        Returns:
            Dictionary of dataset statistics
        """
        if self.ratings_df is None:
            raise ValueError("Data must be loaded first.")

        stats = {}

        # Basic counts
        stats['n_users'] = self.ratings_df['userId'].nunique()
        stats['n_items'] = self.ratings_df['movieId'].nunique()
        stats['n_ratings'] = len(self.ratings_df)

        # Rating distribution
        stats['rating_distribution'] = self.ratings_df['rating'].value_counts().sort_index().to_dict()
        stats['mean_rating'] = self.ratings_df['rating'].mean()
        stats['std_rating'] = self.ratings_df['rating'].std()

        # User statistics
        user_rating_counts = self.ratings_df['userId'].value_counts()
        stats['mean_ratings_per_user'] = user_rating_counts.mean()
        stats['std_ratings_per_user'] = user_rating_counts.std()
        stats['min_ratings_per_user'] = user_rating_counts.min()
        stats['max_ratings_per_user'] = user_rating_counts.max()

        # Item statistics
        item_rating_counts = self.ratings_df['movieId'].value_counts()
        stats['mean_ratings_per_item'] = item_rating_counts.mean()
        stats['std_ratings_per_item'] = item_rating_counts.std()
        stats['min_ratings_per_item'] = item_rating_counts.min()
        stats['max_ratings_per_item'] = item_rating_counts.max()

        # Sparsity
        stats['sparsity'] = 1 - (stats['n_ratings'] / (stats['n_users'] * stats['n_items']))

        return stats

    def save_processed_data(self, filepath: str):
        """Save processed data for reproducibility"""
        if self.ratings_df is None:
            raise ValueError("No processed data to save.")

        self.ratings_df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to: {filepath}")

def load_movielens_data(config=None, dataset_name: str = "ml-latest-small",
                       auto_download: bool = True):
    """
    Convenience function to load and preprocess MovieLens data

    Args:
        config: Configuration object
        dataset_name: Name of MovieLens dataset to use
        auto_download: Whether to automatically download if not found

    Returns:
        Tuple of (loader, train_df, test_df, user_item_matrix)
    """
    loader = MovieLensLoader(config)
    loader.load_data(dataset_name, auto_download)
    loader.preprocess_data()
    loader.create_user_item_matrix()
    train_df, test_df = loader.train_test_split()

    return loader, train_df, test_df, loader.user_item_matrix