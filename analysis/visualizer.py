"""
Academic-standard visualization framework for MovieLens CF analysis
Inspired by VMamba project's elegant academic visualization style
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Set
import os
from scipy.sparse import csr_matrix
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

from config import cfg
from utils.logger import get_logger
from analysis.hyperparameter_optimizer import HyperparameterSearchResult

logger = get_logger("Visualizer")

class AcademicVisualizer:
    """
    Academic-standard visualization suite for collaborative filtering analysis

    Provides publication-ready plots and interactive visualizations for:
    - Dataset analysis and statistics
    - Model performance comparison
    - Similarity analysis
    - Recommendation quality assessment
    """

    def __init__(self, config=None):
        """
        Initialize visualizer with VMamba-inspired academic styling

        Args:
            config: Configuration object
        """
        self.config = config or cfg
        self._setup_academic_style()

    def _setup_academic_style(self):
        """Setup VMamba-inspired academic styling for plots"""
        # VMamba-inspired color palette - sophisticated and academic
        self.vmamba_colors = {
            'primary': '#2E5BBA',      # Professional blue
            'secondary': '#8B4E9C',    # Elegant purple
            'accent': '#E67E22',       # Warm orange
            'success': '#27AE60',      # Professional green
            'warning': '#F39C12',      # Academic gold
            'error': '#E74C3C',        # Attention red
            'neutral_dark': '#2C3E50', # Dark slate
            'neutral_light': '#ECF0F1' # Light gray
        }

        # Create academic color palette
        self.color_palette = [
            self.vmamba_colors['primary'],
            self.vmamba_colors['accent'],
            self.vmamba_colors['success'],
            self.vmamba_colors['secondary'],
            self.vmamba_colors['warning'],
            self.vmamba_colors['error'],
            self.vmamba_colors['neutral_dark']
        ]

        # Performance color scheme (inspired by VMamba performance charts)
        self.performance_colors = {
            'excellent': '#27AE60',    # Green for best performance
            'good': '#2E5BBA',         # Blue for good performance
            'average': '#F39C12',      # Orange for average
            'poor': '#E74C3C'          # Red for poor performance
        }

        # Set academic-standard matplotlib parameters
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 10,
            'figure.titlesize': 15,
            'figure.titleweight': 'bold',
            'figure.dpi': self.config.visualization.dpi,
            'savefig.dpi': self.config.visualization.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.15,
            'axes.grid': True,
            'axes.axisbelow': True,
            'axes.edgecolor': '#2C3E50',
            'axes.linewidth': 0.8,
            'grid.color': '#BDC3C7',
            'grid.linewidth': 0.5,
            'xtick.color': '#2C3E50',
            'ytick.color': '#2C3E50',
            'text.color': '#2C3E50'
        })

        # Create custom colormaps
        self._create_custom_colormaps()

    def _create_custom_colormaps(self):
        """Create custom colormaps inspired by VMamba visualizations"""
        # Academic heatmap colormap
        colors_heatmap = ['#ECF0F1', '#3498DB', '#2E5BBA', '#1B4F72']
        self.academic_cmap = LinearSegmentedColormap.from_list('academic', colors_heatmap)

        # Performance gradient colormap
        colors_perf = [self.performance_colors['poor'],
                      self.performance_colors['average'],
                      self.performance_colors['good'],
                      self.performance_colors['excellent']]
        self.performance_cmap = LinearSegmentedColormap.from_list('performance', colors_perf)

    def plot_dataset_statistics(self, stats: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive dataset statistics with VMamba-inspired styling

        Args:
            stats: Dataset statistics dictionary
            save_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        logger.info("Creating VMamba-style dataset statistics visualization...")

        # Create figure with professional layout
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.3)

        # Main title with academic styling
        fig.suptitle('MovieLens Dataset Analysis', fontsize=18, fontweight='bold', y=0.95)

        # 1. Dataset Overview (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        basic_stats = ['Users', 'Items', 'Ratings']
        values = [stats['n_users'], stats['n_items'], stats['n_ratings']]

        bars = ax1.bar(basic_stats, values, color=self.color_palette[:3],
                      edgecolor='white', linewidth=1.5, alpha=0.8)
        ax1.set_title('Dataset Overview', fontweight='bold', pad=15)
        ax1.set_ylabel('Count', fontweight='bold')

        # Add elegant value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold')

        # 2. Sparsity Analysis (academic style pie chart)
        ax2 = fig.add_subplot(gs[0, 2:])
        sparsity = stats['sparsity']
        density = 1 - sparsity

        colors = [self.vmamba_colors['success'], self.vmamba_colors['neutral_light']]
        wedges, texts, autotexts = ax2.pie([density, sparsity],
                                          labels=['Density', 'Sparsity'],
                                          autopct='%1.2f%%',
                                          colors=colors,
                                          startangle=90,
                                          textprops={'fontweight': 'bold'})
        ax2.set_title('Matrix Sparsity Analysis', fontweight='bold', pad=15)

        # 3. Rating Distribution (elegant histogram)
        ax3 = fig.add_subplot(gs[1, :2])
        rating_dist = stats['rating_distribution']
        ratings = list(rating_dist.keys())
        counts = list(rating_dist.values())

        bars = ax3.bar(ratings, counts, color=self.vmamba_colors['primary'],
                      edgecolor='white', linewidth=1, alpha=0.8)
        ax3.set_title('Rating Distribution', fontweight='bold', pad=15)
        ax3.set_xlabel('Rating Score', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')

        # Add trend line
        z = np.polyfit(ratings, counts, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(ratings), max(ratings), 100)
        ax3.plot(x_smooth, p(x_smooth), color=self.vmamba_colors['accent'],
                linewidth=2, alpha=0.7, linestyle='--')

        # 4. User Activity Analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        user_metrics = ['Mean', 'Std Dev', 'Min', 'Max']
        user_values = [stats['mean_ratings_per_user'], stats['std_ratings_per_user'],
                      stats['min_ratings_per_user'], stats['max_ratings_per_user']]

        bars = ax4.bar(user_metrics, user_values, color=self.color_palette[1],
                      edgecolor='white', linewidth=1.5, alpha=0.8)
        ax4.set_title('User Activity Statistics', fontweight='bold', pad=15)
        ax4.set_ylabel('Ratings per User', fontweight='bold')

        for bar, value in zip(bars, user_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

        # 5. Academic Summary Panel (VMamba-style information panel)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        # Create information boxes similar to VMamba performance tables
        info_data = [
            ('Dataset Statistics', [
                f"Total Users: {stats['n_users']:,}",
                f"Total Items: {stats['n_items']:,}",
                f"Total Ratings: {stats['n_ratings']:,}",
                f"Density: {(1-sparsity)*100:.2f}%"
            ]),
            ('Rating Analysis', [
                f"Mean Rating: {stats['mean_rating']:.3f}",
                f"Rating Std: {stats['std_rating']:.3f}",
                f"Min Rating: {min(rating_dist.keys()):.1f}",
                f"Max Rating: {max(rating_dist.keys()):.1f}"
            ]),
            ('User Behavior', [
                f"Avg Ratings/User: {stats['mean_ratings_per_user']:.1f}",
                f"Most Active User: {stats['max_ratings_per_user']} ratings",
                f"User Std Dev: {stats['std_ratings_per_user']:.1f}",
                f"Min User Activity: {stats['min_ratings_per_user']}"
            ]),
            ('Item Popularity', [
                f"Avg Ratings/Item: {stats['mean_ratings_per_item']:.1f}",
                f"Most Popular Item: {stats['max_ratings_per_item']} ratings",
                f"Item Std Dev: {stats['std_ratings_per_item']:.1f}",
                f"Min Item Ratings: {stats['min_ratings_per_item']}"
            ])
        ]

        # Create elegant information boxes
        box_width = 0.22
        box_height = 0.8
        y_pos = 0.1

        for i, (title, items) in enumerate(info_data):
            x_pos = 0.02 + i * 0.24

            # Create background rectangle
            rect = Rectangle((x_pos, y_pos), box_width, box_height,
                           facecolor=self.vmamba_colors['neutral_light'],
                           edgecolor=self.vmamba_colors['primary'],
                           linewidth=1.5, alpha=0.3)
            ax5.add_patch(rect)

            # Add title
            ax5.text(x_pos + box_width/2, y_pos + box_height - 0.1, title,
                    ha='center', va='top', fontweight='bold', fontsize=12,
                    color=self.vmamba_colors['primary'])

            # Add items
            for j, item in enumerate(items):
                ax5.text(x_pos + 0.01, y_pos + box_height - 0.25 - j*0.12, item,
                        ha='left', va='top', fontsize=10,
                        color=self.vmamba_colors['neutral_dark'])

        if save_path:
            self._save_figure(fig, save_path, 'dataset_statistics')

        return fig

    def plot_model_comparison(self, results: Dict[str, Dict],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        VMamba-style model performance comparison

        Args:
            results: Dictionary mapping model names to their evaluation results
            save_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        logger.info("Creating VMamba-style model comparison visualization...")

        # Extract model data
        models = list(results.keys())

        # Create figure with professional layout
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

        # Main title
        fig.suptitle('Collaborative Filtering Models Performance Analysis',
                    fontsize=18, fontweight='bold', y=0.96)

        # 1. Performance Overview (VMamba-style performance table)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')

        # Extract key metrics
        performance_data = []
        for model in models:
            rating_pred = results[model].get('rating_prediction', {})
            timing = results[model].get('timing', {})

            mae = rating_pred.get('mae', 0)
            rmse = rating_pred.get('rmse', 0)
            corr = rating_pred.get('pearson_correlation', 0)
            train_time = timing.get('training_time', 0)

            # Determine performance level based on MAE (lower is better)
            if mae < 0.5:
                perf_color = self.performance_colors['excellent']
                perf_symbol = '★'
            elif mae < 0.6:
                perf_color = self.performance_colors['good']
                perf_symbol = '▲'
            elif mae < 0.7:
                perf_color = self.performance_colors['average']
                perf_symbol = '●'
            else:
                perf_color = self.performance_colors['poor']
                perf_symbol = '▼'

            performance_data.append([model, mae, rmse, corr, train_time, perf_color, perf_symbol])

        # Create VMamba-style performance table
        y_start = 0.8
        col_widths = [0.2, 0.15, 0.15, 0.15, 0.15, 0.2]
        headers = ['Model', 'MAE ↓', 'RMSE ↓', 'Correlation ↑', 'Time (s)', 'Performance']

        # Draw headers
        x_pos = 0.05
        for i, (header, width) in enumerate(zip(headers, col_widths)):
            rect = Rectangle((x_pos, y_start), width, 0.1,
                           facecolor=self.vmamba_colors['primary'],
                           alpha=0.8)
            ax1.add_patch(rect)
            ax1.text(x_pos + width/2, y_start + 0.05, header,
                    ha='center', va='center', fontweight='bold',
                    color='white', fontsize=11)
            x_pos += width

        # Draw data rows
        for i, (model, mae, rmse, corr, time, color, symbol) in enumerate(performance_data):
            y_pos = y_start - (i + 1) * 0.08
            x_pos = 0.05

            # Alternate row colors
            row_color = self.vmamba_colors['neutral_light'] if i % 2 == 0 else 'white'

            # Model name
            rect = Rectangle((x_pos, y_pos), col_widths[0], 0.08,
                           facecolor=row_color, alpha=0.5,
                           edgecolor=self.vmamba_colors['primary'], linewidth=0.5)
            ax1.add_patch(rect)
            ax1.text(x_pos + col_widths[0]/2, y_pos + 0.04, model,
                    ha='center', va='center', fontweight='bold', fontsize=10)
            x_pos += col_widths[0]

            # Metrics
            values = [f'{mae:.3f}', f'{rmse:.3f}', f'{corr:.3f}', f'{time:.3f}']
            for j, (value, width) in enumerate(zip(values, col_widths[1:5])):
                rect = Rectangle((x_pos, y_pos), width, 0.08,
                               facecolor=row_color, alpha=0.5,
                               edgecolor=self.vmamba_colors['primary'], linewidth=0.5)
                ax1.add_patch(rect)
                ax1.text(x_pos + width/2, y_pos + 0.04, value,
                        ha='center', va='center', fontsize=10)
                x_pos += width

            # Performance indicator
            rect = Rectangle((x_pos, y_pos), col_widths[5], 0.08,
                           facecolor=color, alpha=0.6,
                           edgecolor=self.vmamba_colors['primary'], linewidth=0.5)
            ax1.add_patch(rect)
            ax1.text(x_pos + col_widths[5]/2, y_pos + 0.04, symbol,
                    ha='center', va='center', fontweight='bold',
                    color='white', fontsize=14)

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title('Model Performance Summary', fontweight='bold', fontsize=14, pad=20)

        # 2. Error Metrics Comparison (elegant bar chart)
        ax2 = fig.add_subplot(gs[1, :2])
        mae_values = [results[model]['rating_prediction'].get('mae', 0) for model in models]
        rmse_values = [results[model]['rating_prediction'].get('rmse', 0) for model in models]

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax2.bar(x - width/2, mae_values, width, label='MAE',
                       color=self.vmamba_colors['primary'], alpha=0.8,
                       edgecolor='white', linewidth=1.5)
        bars2 = ax2.bar(x + width/2, rmse_values, width, label='RMSE',
                       color=self.vmamba_colors['accent'], alpha=0.8,
                       edgecolor='white', linewidth=1.5)

        ax2.set_xlabel('Models', fontweight='bold')
        ax2.set_ylabel('Error Value', fontweight='bold')
        ax2.set_title('Prediction Error Comparison', fontweight='bold', pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend(frameon=True, fancybox=True, shadow=True)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.3f}', ha='center', va='bottom',
                        fontweight='bold', fontsize=9)

        # 3. Correlation Analysis (radar-style)
        ax3 = fig.add_subplot(gs[1, 2])
        corr_values = [models_results[m].get('rating_prediction', {}).get('pearson_correlation', 0) for m in models]
        fig_corr, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x, corr_values, color=self.vmamba_colors['success'], edgecolor='white', linewidth=1.0)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Correlation')
        ax.set_title('Correlation Analysis', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        figures['model_correlation'] = fig_corr

        train_times = [results[model].get('timing', {}).get('training_time', 0) for model in models]
        eval_times = [results[model].get('timing', {}).get('evaluation_time', 0) for model in models]

        bars1 = ax4.bar(x - width/2, train_times, width, label='Training Time',
                       color=self.vmamba_colors['success'], alpha=0.8,
                       edgecolor='white', linewidth=1.5)
        bars2 = ax4.bar(x + width/2, eval_times, width, label='Evaluation Time',
                       color=self.vmamba_colors['warning'], alpha=0.8,
                       edgecolor='white', linewidth=1.5)

        ax4.set_xlabel('Models', fontweight='bold')
        ax4.set_ylabel('Time (seconds)', fontweight='bold')
        ax4.set_title('Computational Efficiency Analysis', fontweight='bold', pad=15)
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.legend(frameon=True, fancybox=True, shadow=True)

        # 5. Best Performance Highlight
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis('off')

        # Find best performing model
        best_mae = min(mae_values)
        best_model = models[mae_values.index(best_mae)]
        best_corr = corr_values[mae_values.index(best_mae)]

        # Create highlight box
        rect = Rectangle((0.1, 0.1), 0.8, 0.8,
                        facecolor=self.performance_colors['excellent'],
                        alpha=0.2, edgecolor=self.performance_colors['excellent'],
                        linewidth=2)
        ax5.add_patch(rect)

        ax5.text(0.5, 0.8, 'Best Model', ha='center', va='center',
                fontweight='bold', fontsize=14,
                color=self.performance_colors['excellent'])

        ax5.text(0.5, 0.6, best_model, ha='center', va='center',
                fontweight='bold', fontsize=16,
                color=self.vmamba_colors['primary'])

        ax5.text(0.5, 0.4, f'MAE: {best_mae:.3f}', ha='center', va='center',
                fontweight='bold', fontsize=12,
                color=self.vmamba_colors['neutral_dark'])

        ax5.text(0.5, 0.25, f'Correlation: {best_corr:.3f}', ha='center', va='center',
                fontweight='bold', fontsize=12,
                color=self.vmamba_colors['neutral_dark'])

        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)

        if save_path:
            self._save_figure(fig, save_path, 'model_comparison')

        return fig

    def plot_similarity_heatmap(self, similarity_matrix: np.ndarray,
                               title: str = "Similarity Matrix",
                               sample_size: int = 100,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot similarity matrix heatmap

        Args:
            similarity_matrix: Similarity matrix to visualize
            title: Plot title
            sample_size: Number of entities to sample for visualization
            save_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        logger.info(f"Creating similarity heatmap: {title}")

        # Sample for visualization if matrix is too large
        if similarity_matrix.shape[0] > sample_size:
            indices = np.random.choice(similarity_matrix.shape[0], sample_size, replace=False)
            sampled_matrix = similarity_matrix[np.ix_(indices, indices)]
        else:
            sampled_matrix = similarity_matrix

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(sampled_matrix, cmap='viridis', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Similarity Score')

        ax.set_title(title)
        ax.set_xlabel('Entity Index')
        ax.set_ylabel('Entity Index')

        if save_path:
            self._save_figure(fig, save_path, f'similarity_heatmap_{title.lower().replace(" ", "_")}')

        return fig

    def plot_rating_distribution_analysis(self, ratings_data: pd.DataFrame,
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Detailed analysis of rating distributions

        Args:
            ratings_data: DataFrame with rating data
            save_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        logger.info("Creating rating distribution analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Rating Distribution Analysis', fontsize=16, fontweight='bold')

        # 1. Overall rating distribution
        ax = axes[0, 0]
        ratings_data['rating'].hist(bins=20, ax=ax, color=self.color_palette[0], alpha=0.7)
        ax.set_title('Overall Rating Distribution')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')

        # 2. Rating distribution by user activity
        ax = axes[0, 1]
        user_activity = ratings_data.groupby('userId').size()

        # Categorize users by activity
        low_activity = user_activity[user_activity <= user_activity.quantile(0.33)]
        med_activity = user_activity[(user_activity > user_activity.quantile(0.33)) &
                                   (user_activity <= user_activity.quantile(0.67))]
        high_activity = user_activity[user_activity > user_activity.quantile(0.67)]

        low_ratings = ratings_data[ratings_data['userId'].isin(low_activity.index)]['rating']
        med_ratings = ratings_data[ratings_data['userId'].isin(med_activity.index)]['rating']
        high_ratings = ratings_data[ratings_data['userId'].isin(high_activity.index)]['rating']

        ax.hist([low_ratings, med_ratings, high_ratings],
               bins=10, label=['Low Activity', 'Medium Activity', 'High Activity'],
               color=self.color_palette[:3], alpha=0.7)
        ax.set_title('Rating Distribution by User Activity')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        ax.legend()

        # 3. Rating distribution by item popularity
        ax = axes[1, 0]
        item_popularity = ratings_data.groupby('movieId').size()

        # Categorize items by popularity
        unpopular = item_popularity[item_popularity <= item_popularity.quantile(0.33)]
        moderate = item_popularity[(item_popularity > item_popularity.quantile(0.33)) &
                                 (item_popularity <= item_popularity.quantile(0.67))]
        popular = item_popularity[item_popularity > item_popularity.quantile(0.67)]

        unpopular_ratings = ratings_data[ratings_data['movieId'].isin(unpopular.index)]['rating']
        moderate_ratings = ratings_data[ratings_data['movieId'].isin(moderate.index)]['rating']
        popular_ratings = ratings_data[ratings_data['movieId'].isin(popular.index)]['rating']

        ax.hist([unpopular_ratings, moderate_ratings, popular_ratings],
               bins=10, label=['Unpopular', 'Moderate', 'Popular'],
               color=self.color_palette[3:6], alpha=0.7)
        ax.set_title('Rating Distribution by Item Popularity')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        ax.legend()

        # 4. Box plot of ratings
        ax = axes[1, 1]
        ratings_data.boxplot(column='rating', ax=ax)
        ax.set_title('Rating Distribution Box Plot')
        ax.set_ylabel('Rating')

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path, 'rating_distribution_analysis')

        return fig

    def plot_performance_metrics(self, metrics_data: Dict[str, Dict],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot detailed performance metrics analysis

        Args:
            metrics_data: Dictionary containing metrics for different k values
            save_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        logger.info("Creating performance metrics visualization...")

        if not metrics_data:
            logger.warning("No metrics data provided")
            return plt.figure()

        k_values = sorted(list(metrics_data.keys()))
        metrics = ['precision', 'recall', 'f1', 'ndcg', 'hit_rate']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Metrics Analysis', fontsize=16, fontweight='bold')

        # Plot each metric
        for i, metric in enumerate(metrics):
            if i < 5:  # We have 5 subplots
                row = i // 3
                col = i % 3
                ax = axes[row, col]

                values = [metrics_data[k].get(metric, 0) for k in k_values]

                ax.plot(k_values, values, marker='o', linewidth=2,
                       color=self.color_palette[i], markersize=8)
                ax.set_title(f'{metric.upper()}@K')
                ax.set_xlabel('K (Number of Recommendations)')
                ax.set_ylabel(f'{metric.upper()} Score')
                ax.grid(True, alpha=0.3)

                # Add value labels
                for k, value in zip(k_values, values):
                    ax.annotate(f'{value:.3f}', (k, value),
                              textcoords="offset points", xytext=(0,10), ha='center')

        # Summary comparison
        ax = axes[1, 2]
        x = np.arange(len(k_values))
        width = 0.15

        for i, metric in enumerate(metrics):
            values = [metrics_data[k].get(metric, 0) for k in k_values]
            ax.bar(x + i*width, values, width, label=metric.upper(),
                  color=self.color_palette[i])

        ax.set_xlabel('K Values')
        ax.set_ylabel('Score')
        ax.set_title('Metrics Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([f'K={k}' for k in k_values])
        ax.legend()

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path, 'performance_metrics')

        return fig

    def create_interactive_dashboard(self, results: Dict[str, Dict],
                                   save_path: Optional[str] = None) -> str:
        """
        Create interactive dashboard using Plotly

        Args:
            results: Comprehensive results dictionary
            save_path: Path to save the HTML file

        Returns:
            Path to saved HTML file
        """
        logger.info("Creating interactive dashboard...")

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Model Comparison (MAE)', 'Model Comparison (RMSE)',
                          'Ranking Metrics', 'Correlation Analysis',
                          'Performance Trends', 'Summary Statistics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )

        models = list(results.keys())

        # MAE comparison
        mae_values = [results[model]['rating_prediction'].get('mae', 0) for model in models]
        fig.add_trace(
            go.Bar(x=models, y=mae_values, name='MAE', marker_color='blue'),
            row=1, col=1
        )

        # RMSE comparison
        rmse_values = [results[model]['rating_prediction'].get('rmse', 0) for model in models]
        fig.add_trace(
            go.Bar(x=models, y=rmse_values, name='RMSE', marker_color='orange'),
            row=1, col=2
        )

        # Ranking metrics (if available)
        if any('ranking' in results[model] for model in models):
            precision_10 = [results[model]['ranking'].get(10, {}).get('precision', 0) for model in models]
            recall_10 = [results[model]['ranking'].get(10, {}).get('recall', 0) for model in models]

            fig.add_trace(
                go.Scatter(x=models, y=precision_10, name='Precision@10', mode='markers+lines'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=models, y=recall_10, name='Recall@10', mode='markers+lines'),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title_text="MovieLens CF Model Analysis Dashboard",
            showlegend=True,
            height=900
        )

        # Save as HTML
        if save_path and save_path.endswith('.html'):
            dashboard_file = save_path
            os.makedirs(os.path.dirname(dashboard_file) or '.', exist_ok=True)
        else:
            target_dir = save_path or self.config.experiment.plots_dir
            os.makedirs(target_dir, exist_ok=True)
            dashboard_file = os.path.join(target_dir, 'interactive_dashboard.html')

        fig.write_html(dashboard_file)

        logger.info(f"Interactive dashboard saved to: {dashboard_file}")
        return dashboard_file

    def _save_figure(self, fig: plt.Figure, base_path: str, filename: str) -> List[str]:
        """Save figure in multiple formats and return saved paths"""
        saved_paths: List[str] = []
        if not self.config.visualization.save_plots:
            return saved_paths

        os.makedirs(base_path, exist_ok=True)

        for fmt in self.config.visualization.plot_formats:
            filepath = os.path.join(base_path, f"{filename}.{fmt}")
            try:
                fig.savefig(filepath, format=fmt, dpi=self.config.visualization.dpi)
                logger.info(f"Plot saved: {filepath}")
                saved_paths.append(filepath)
            except Exception as e:
                logger.error(f"Failed to save plot {filepath}: {e}")
        return saved_paths



    def _create_dataset_figures(self, stats: Dict) -> Dict[str, plt.Figure]:
        figures: Dict[str, plt.Figure] = {}

        overview_labels = ['Users', 'Items', 'Ratings']
        overview_values = [
            stats.get('n_users', 0),
            stats.get('n_items', 0),
            stats.get('n_ratings', 0)
        ]
        fig_overview, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(overview_labels, overview_values, color=self.color_palette[:3], edgecolor='white', linewidth=1.2)
        ax.set_title('Dataset Overview', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        for bar, value in zip(bars, overview_values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value):,}", ha='center', va='bottom', fontweight='bold')
        figures['dataset_overview'] = fig_overview

        fig_sparsity, ax = plt.subplots(figsize=(5, 5))
        sparsity = stats.get('sparsity', 0)
        density = max(0.0, 1 - sparsity)
        ax.pie([density, sparsity], labels=['Density', 'Sparsity'], autopct='%1.2f%%', startangle=90,
               colors=[self.vmamba_colors['success'], self.vmamba_colors['neutral_light']], textprops={'fontweight': 'bold'})
        ax.set_title('Matrix Sparsity', fontweight='bold')
        figures['dataset_sparsity'] = fig_sparsity

        rating_distribution = stats.get('rating_distribution', {}) or {}
        ratings = sorted(rating_distribution.keys())
        counts = [rating_distribution.get(r, 0) for r in ratings]
        fig_rating, ax = plt.subplots(figsize=(8, 5))
        ax.bar(ratings, counts, color=self.vmamba_colors['primary'], edgecolor='white', linewidth=1.0, alpha=0.85)
        ax.set_title('Rating Distribution', fontweight='bold')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        figures['rating_distribution'] = fig_rating

        user_metrics = {
            'Mean': stats.get('mean_ratings_per_user', 0),
            'Std Dev': stats.get('std_ratings_per_user', 0),
            'Min': stats.get('min_ratings_per_user', 0),
            'Max': stats.get('max_ratings_per_user', 0)
        }
        fig_user, ax = plt.subplots(figsize=(8, 5))
        ax.bar(user_metrics.keys(), user_metrics.values(), color=self.vmamba_colors['accent'], edgecolor='white', linewidth=1.0)
        ax.set_title('User Activity Statistics', fontweight='bold')
        ax.set_ylabel('Ratings Count')
        figures['user_activity'] = fig_user

        item_metrics = {
            'Mean': stats.get('mean_ratings_per_item', 0),
            'Std Dev': stats.get('std_ratings_per_item', 0),
            'Min': stats.get('min_ratings_per_item', 0),
            'Max': stats.get('max_ratings_per_item', 0)
        }
        fig_item, ax = plt.subplots(figsize=(8, 5))
        ax.bar(item_metrics.keys(), item_metrics.values(), color=self.vmamba_colors['secondary'], edgecolor='white', linewidth=1.0)
        ax.set_title('Item Popularity Statistics', fontweight='bold')
        ax.set_ylabel('Ratings Count')
        figures['item_activity'] = fig_item

        return figures

    def _create_model_performance_figures(self, models_results: Dict[str, Dict]) -> Dict[str, plt.Figure]:
        figures: Dict[str, plt.Figure] = {}
        if not models_results:
            return figures

        models = list(models_results.keys())
        x = np.arange(len(models))

        mae_values = [models_results[m].get('rating_prediction', {}).get('mae', 0) for m in models]
        rmse_values = [models_results[m].get('rating_prediction', {}).get('rmse', 0) for m in models]
        fig_error, ax = plt.subplots(figsize=(9, 5))
        width = 0.35
        ax.bar(x - width / 2, mae_values, width, label='MAE', color=self.vmamba_colors['primary'], edgecolor='white', linewidth=1.0)
        ax.bar(x + width / 2, rmse_values, width, label='RMSE', color=self.vmamba_colors['accent'], edgecolor='white', linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.set_ylabel('Error')
        ax.set_title('Rating Prediction Errors', fontweight='bold')
        ax.legend()
        figures['model_error'] = fig_error

        corr_values = [models_results[m].get('rating_prediction', {}).get('pearson_correlation', 0) for m in models]
        fig_corr, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x, corr_values, color=self.vmamba_colors['success'], edgecolor='white', linewidth=1.0)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Correlation')
        ax.set_title('Correlation Analysis', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        figures['model_correlation'] = fig_corr

        train_times = [models_results[m].get('timing', {}).get('training_time', 0) for m in models]
        eval_times = [models_results[m].get('timing', {}).get('evaluation_time', 0) for m in models]
        fig_time, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width / 2, train_times, width, label='Training', color=self.vmamba_colors['success'], edgecolor='white', linewidth=1.0)
        ax.bar(x + width / 2, eval_times, width, label='Evaluation', color=self.vmamba_colors['warning'], edgecolor='white', linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.set_ylabel('Seconds')
        ax.set_title('Computation Time', fontweight='bold')
        ax.legend()
        figures['model_timing'] = fig_time

        ranking_available = any('ranking' in models_results[m] for m in models)
        if ranking_available:
            fig_rank, ax = plt.subplots(figsize=(9, 5))
            precision = [models_results[m].get('ranking', {}).get(10, {}).get('precision', 0) for m in models]
            recall = [models_results[m].get('ranking', {}).get(10, {}).get('recall', 0) for m in models]
            ax.plot(x, precision, marker='o', label='Precision@10', color=self.vmamba_colors['primary'])
            ax.plot(x, recall, marker='s', label='Recall@10', color=self.vmamba_colors['accent'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Score')
            ax.set_title('Ranking Metrics (k=10)', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=30, ha='right')
            ax.legend()
            figures['model_ranking'] = fig_rank

        return figures

    def _resolve_temporal_pairs(self, models_results: Dict[str, Dict]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for model_name in models_results.keys():
            if not model_name.lower().startswith('temporal'):
                continue
            base_candidate = model_name[len('Temporal'):]
            if base_candidate.startswith('_'):
                base_candidate = base_candidate[1:]
            if base_candidate in models_results:
                pairs.append((base_candidate, model_name))
        return pairs

    def _create_temporal_visualizations(self, loader, models_results: Dict[str, Dict]) -> Dict[str, plt.Figure]:
        figures: Dict[str, plt.Figure] = {}
        pairs = self._resolve_temporal_pairs(models_results)
        if not pairs:
            return figures

        improvement_fig = self._plot_temporal_improvements(pairs, models_results)
        if improvement_fig is not None:
            figures['temporal_improvement'] = improvement_fig

        heatmap_fig = self._plot_recency_heatmap(loader)
        if heatmap_fig is not None:
            figures['temporal_recency_heatmap'] = heatmap_fig

        timeline_fig = self._plot_temporal_timeline(loader, pairs, models_results)
        if timeline_fig is not None:
            figures['temporal_timeline'] = timeline_fig

        return figures

    def _plot_temporal_improvements(self, pairs: List[Tuple[str, str]],
                                     models_results: Dict[str, Dict]) -> Optional[plt.Figure]:
        if not pairs:
            return None

        metric_defs = [('mae', 'MAE'), ('rmse', 'RMSE')]
        primary_k = cfg.evaluation.top_k_recommendations[0] if cfg.evaluation.top_k_recommendations else 10
        metric_defs.append((f'precision@{primary_k}', f'Precision@{primary_k}'))

        baseline_values: Dict[str, List[float]] = {key: [] for key, _ in metric_defs}
        temporal_values: Dict[str, List[float]] = {key: [] for key, _ in metric_defs}

        for base_name, temporal_name in pairs:
            base_result = models_results.get(base_name, {})
            temporal_result = models_results.get(temporal_name, {})

            base_ratings = base_result.get('rating_prediction', {})
            temporal_ratings = temporal_result.get('rating_prediction', {})

            baseline_values['mae'].append(float(base_ratings.get('mae', np.nan)))
            temporal_values['mae'].append(float(temporal_ratings.get('mae', np.nan)))

            baseline_values['rmse'].append(float(base_ratings.get('rmse', np.nan)))
            temporal_values['rmse'].append(float(temporal_ratings.get('rmse', np.nan)))

            base_ranking = base_result.get('ranking', {}).get(primary_k, {})
            temporal_ranking = temporal_result.get('ranking', {}).get(primary_k, {})
            baseline_values[f'precision@{primary_k}'].append(float(base_ranking.get('precision', np.nan)))
            temporal_values[f'precision@{primary_k}'].append(float(temporal_ranking.get('precision', np.nan)))

        if not baseline_values['mae']:
            return None

        fig, axes = plt.subplots(1, len(metric_defs), figsize=(14, 4), constrained_layout=True)
        if len(metric_defs) == 1:
            axes = [axes]

        x = np.arange(len(pairs))
        width = 0.35

        for ax, (metric_key, metric_label) in zip(axes, metric_defs):
            base_vals = np.array(baseline_values[metric_key])
            temporal_vals = np.array(temporal_values[metric_key])

            ax.bar(x - width / 2, base_vals, width, label='Baseline',
                   color=self.vmamba_colors['primary'], edgecolor='white', linewidth=1.0, alpha=0.85)
            ax.bar(x + width / 2, temporal_vals, width, label='Temporal',
                   color=self.vmamba_colors['accent'], edgecolor='white', linewidth=1.0, alpha=0.85)

            improvements = base_vals - temporal_vals
            for idx, delta in enumerate(improvements):
                if np.isnan(delta):
                    continue
                y_pos = max(base_vals[idx], temporal_vals[idx])
                ax.text(idx, y_pos + 0.01 * (abs(y_pos) + 1), f"Δ {delta:.3f}",
                        ha='center', va='bottom', fontsize=8, color=self.vmamba_colors['neutral_dark'])

            ax.set_title(metric_label, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([temporal for _, temporal in pairs], rotation=30, ha='right')
            ax.grid(alpha=0.3)

        axes[0].set_ylabel('Metric value')
        axes[0].legend(loc='upper right')
        fig.suptitle('Temporal CF Improvements over Baselines', fontsize=16, fontweight='bold')
        return fig

    def _plot_recency_heatmap(self, loader) -> Optional[plt.Figure]:
        if loader is None:
            return None

        timestamp_matrix = getattr(loader, 'timestamp_matrix', None)
        if timestamp_matrix is None or timestamp_matrix.nnz == 0:
            return None

        user_summary = getattr(loader, 'user_recency_summary', {}) or {}
        item_summary = getattr(loader, 'item_recency_summary', {}) or {}
        user_counts = user_summary.get('counts')
        item_counts = item_summary.get('counts')

        if user_counts is None or item_counts is None:
            return None

        n_users, n_items = timestamp_matrix.shape
        sample_users = min(30, n_users)
        sample_items = min(30, n_items)
        if sample_users == 0 or sample_items == 0:
            return None

        top_user_indices = np.argsort(user_counts)[-sample_users:]
        top_item_indices = np.argsort(item_counts)[-sample_items:]

        submatrix = timestamp_matrix[top_user_indices][:, top_item_indices].toarray()
        rating_mask_matrix = getattr(loader, 'user_item_matrix', None)
        if rating_mask_matrix is not None:
            rating_mask = rating_mask_matrix[top_user_indices][:, top_item_indices].toarray() > 0
        else:
            rating_mask = submatrix > 0

        temporal_stats = getattr(loader, 'temporal_statistics', {}) or {}
        latest_time = temporal_stats.get('global_latest_timestamp')
        if latest_time is None:
            latest_time = float(np.max(submatrix)) if submatrix.size else 0.0

        recency = latest_time - submatrix
        recency[~rating_mask] = np.nan

        if np.all(np.isnan(recency)):
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(recency, ax=ax, cmap=self.academic_cmap, cbar_kws={'label': 'Days since interaction'},
                    mask=np.isnan(recency))
        user_labels = [str(getattr(loader, 'inverse_user_mapping', {}).get(int(idx), idx)) for idx in top_user_indices]
        item_labels = [str(getattr(loader, 'inverse_item_mapping', {}).get(int(idx), idx)) for idx in top_item_indices]
        ax.set_xticks(np.arange(len(item_labels)) + 0.5)
        ax.set_xticklabels(item_labels, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(user_labels)) + 0.5)
        ax.set_yticklabels(user_labels)
        ax.set_title('Recency Landscape of Active Users/Items', fontweight='bold')
        ax.set_xlabel('Items')
        ax.set_ylabel('Users')
        fig.tight_layout()
        return fig

    def _plot_temporal_timeline(self, loader, pairs: List[Tuple[str, str]],
                                 models_results: Dict[str, Dict]) -> Optional[plt.Figure]:
        if not pairs:
            return None

        base_name, temporal_name = pairs[0]
        base_details = models_results.get(base_name, {}).get('prediction_details')
        temporal_details = models_results.get(temporal_name, {}).get('prediction_details')

        if not base_details or not temporal_details:
            return None

        base_df = pd.DataFrame(base_details)
        temporal_df = pd.DataFrame(temporal_details)
        if base_df.empty or temporal_df.empty:
            return None

        base_df = base_df.rename(columns={'pred_rating': 'pred_rating_base',
                                          'normalized_timestamp': 'normalized_timestamp_base'})
        temporal_df = temporal_df.rename(columns={'pred_rating': 'pred_rating_temporal',
                                                  'normalized_timestamp': 'normalized_timestamp_temporal'})

        merged = pd.merge(
            base_df,
            temporal_df[['user_id', 'item_id', 'timestamp', 'pred_rating_temporal', 'normalized_timestamp_temporal']],
            on=['user_id', 'item_id', 'timestamp'],
            how='inner'
        )

        if merged.empty:
            return None

        merged['normalized_timestamp'] = merged['normalized_timestamp_base'].fillna(
            merged['normalized_timestamp_temporal']
        )
        merged['abs_error_base'] = np.abs(merged['true_rating'] - merged['pred_rating_base'])
        merged['abs_error_temporal'] = np.abs(merged['true_rating'] - merged['pred_rating_temporal'])

        timeline_values = merged['normalized_timestamp'].dropna()
        if timeline_values.empty:
            return None

        quantiles = np.linspace(0, 1, 6)
        bin_edges = np.unique(np.quantile(timeline_values, quantiles))
        if bin_edges.size < 2:
            min_val, max_val = timeline_values.min(), timeline_values.max()
            if np.isclose(min_val, max_val):
                return None
            bin_edges = np.linspace(min_val, max_val, 6)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        dataset_counts: List[int] = []
        if loader is not None and getattr(loader, 'ratings_df', None) is not None and loader.timestamp_reference is not None:
            ratings_df = loader.ratings_df.copy()
            if 'timestamp' in ratings_df.columns:
                ratings_df['normalized_timestamp'] = (
                    (ratings_df['timestamp'].astype(float) - loader.timestamp_reference) / loader.timestamp_unit
                )
                for start, end in zip(bin_edges[:-1], bin_edges[1:]):
                    mask = (ratings_df['normalized_timestamp'] >= start) & (ratings_df['normalized_timestamp'] < end)
                    dataset_counts.append(int(mask.sum()))
            else:
                dataset_counts = [0] * len(bin_centers)
        else:
            dataset_counts = [0] * len(bin_centers)

        base_mae_bins: List[float] = []
        temporal_mae_bins: List[float] = []
        improvement_bins: List[float] = []

        for start, end in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (merged['normalized_timestamp'] >= start) & (merged['normalized_timestamp'] < end)
            if mask.sum() == 0:
                base_mae_bins.append(np.nan)
                temporal_mae_bins.append(np.nan)
                improvement_bins.append(np.nan)
            else:
                base_mae = float(merged.loc[mask, 'abs_error_base'].mean())
                temporal_mae = float(merged.loc[mask, 'abs_error_temporal'].mean())
                base_mae_bins.append(base_mae)
                temporal_mae_bins.append(temporal_mae)
                improvement_bins.append(base_mae - temporal_mae)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        x = np.arange(len(bin_centers))

        ax1.bar(x, dataset_counts, color=self.vmamba_colors['neutral_light'],
                edgecolor=self.vmamba_colors['neutral_dark'], alpha=0.7)
        ax1.set_ylabel('Ratings count')
        ax1.set_xlabel('Temporal bins')
        ax1.set_title('Temporal Rating Density vs Performance', fontweight='bold')

        ax2 = ax1.twinx()
        ax2.plot(x, base_mae_bins, marker='o', label='Baseline MAE', color=self.vmamba_colors['primary'])
        ax2.plot(x, temporal_mae_bins, marker='s', label='Temporal MAE', color=self.vmamba_colors['accent'])
        ax2.plot(x, improvement_bins, marker='^', linestyle='--', label='MAE Δ', color=self.vmamba_colors['success'])
        ax2.set_ylabel('Error / Improvement')

        if loader is not None and getattr(loader, 'timestamp_reference', None) is not None:
            midpoint_seconds = loader.timestamp_reference + np.array(bin_centers) * loader.timestamp_unit
            labels = pd.to_datetime(midpoint_seconds, unit='s').strftime('%Y-%m')
        else:
            labels = [f"{center:.1f}" for center in bin_centers]

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.grid(alpha=0.3)
        ax2.grid(alpha=0.1)
        ax2.legend(loc='upper right')
        fig.tight_layout()
        return fig

    def generate_all_visualizations(self, loader, models_results: Dict,
                                  save_path: Optional[str] = None) -> Dict[str, str]:
        """
        Generate all visualizations for comprehensive analysis

        Args:
            loader: DataLoader instance with statistics
            models_results: Dictionary of model evaluation results
            save_path: Base path for saving plots

        Returns:
            Dictionary mapping plot names to file paths
        """
        logger.log_phase("Generating All Visualizations")

        save_path = save_path or self.config.experiment.plots_dir
        saved_plots: Dict[str, str] = {}

        try:
            if hasattr(loader, 'get_dataset_statistics'):
                stats = loader.get_dataset_statistics()
                dataset_figures = self._create_dataset_figures(stats)
                for name, fig in dataset_figures.items():
                    paths = self._save_figure(fig, save_path, name)
                    if paths:
                        saved_plots[name] = paths[0]
                    plt.close(fig)

            if models_results:
                model_figures = self._create_model_performance_figures(models_results)
                for name, fig in model_figures.items():
                    paths = self._save_figure(fig, save_path, name)
                    if paths:
                        saved_plots[name] = paths[0]
                    plt.close(fig)

                temporal_figures = self._create_temporal_visualizations(loader, models_results)
                for name, fig in temporal_figures.items():
                    paths = self._save_figure(fig, save_path, name)
                    if paths:
                        saved_plots[name] = paths[0]
                    plt.close(fig)

            if models_results:
                dashboard_path = self.create_interactive_dashboard(models_results, save_path)
                saved_plots['interactive_dashboard'] = dashboard_path

            logger.info(f"Generated {len(saved_plots)} visualizations")

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

        return saved_plots

    def plot_hyperparameter_analysis(self, hp_result: HyperparameterSearchResult,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive hyperparameter analysis visualization

        Args:
            hp_result: Hyperparameter search results
            save_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Main title
        fig.suptitle('Hyperparameter Optimization Analysis',
                    fontsize=16, fontweight='bold', y=0.98)

        # 1. Search convergence plot
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_search_convergence(ax1, hp_result)

        # 2. Best parameters summary
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_best_parameters_summary(ax2, hp_result)

        # 3. Parameter sensitivity analysis
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_parameter_sensitivity(ax3, hp_result)

        # 4. Score distribution
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_score_distribution(ax4, hp_result)

        # 5. Model type comparison
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_model_type_comparison(ax5, hp_result)

        # 6. Statistical significance
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_statistical_analysis(ax6, hp_result)

        # 7. Parameter correlation heatmap
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_parameter_correlation(ax7, hp_result)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path, "hyperparameter_analysis")

        return fig

    def _plot_search_convergence(self, ax, hp_result: HyperparameterSearchResult):
        """Plot search convergence curve"""
        if not hp_result.search_history:
            ax.text(0.5, 0.5, 'No Search History', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return

        iterations = [r['iteration'] for r in hp_result.search_history]
        scores = [r['mean_score'] for r in hp_result.search_history]

        # Plot convergence curve
        ax.plot(iterations, scores, 'o-', color=self.vmamba_colors['primary'],
               linewidth=2, markersize=4, alpha=0.7)

        # Highlight best score
        best_idx = hp_result.convergence_info.get('best_iteration', 0)
        if best_idx < len(scores):
            ax.plot(best_idx, scores[best_idx], 'o', color=self.vmamba_colors['success'],
                   markersize=10, label=f'Best Score: {scores[best_idx]:.4f}')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Score')
        ax.set_title('Hyperparameter Search Convergence', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_best_parameters_summary(self, ax, hp_result: HyperparameterSearchResult):
        """Plot best parameters summary as text box"""
        ax.axis('off')

        # Create elegant parameter summary box
        summary_text = "Best Configuration\n\n"
        if hp_result.best_params:
            for param, value in hp_result.best_params.items():
                summary_text += f"{param.replace('_', ' ').title()}: {value}\n"

        summary_text += f"\nBest Score: {hp_result.best_score:.6f}"
        summary_text += f"\nModel Type: {hp_result.best_model_type.replace('_', ' ').upper()}"

        if getattr(hp_result, 'best_objective_scores', None):
            summary_text += "\n\nObjective Scores\n"
            for objective, value in hp_result.best_objective_scores.items():
                label = objective.replace('_', ' ').title()
                summary_text += f"{label}: {value:.4f}\n"

        # Create styled text box
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor=self.vmamba_colors['neutral_light'],
                         edgecolor=self.vmamba_colors['primary'], linewidth=2, alpha=0.8)

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=bbox_props, family='monospace')

    def _plot_parameter_sensitivity(self, ax, hp_result: HyperparameterSearchResult):
        """Plot parameter sensitivity analysis"""
        if not hp_result.all_results:
            ax.text(0.5, 0.5, 'No Results Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return

        # Extract parameter values and scores
        param_effects = {}
        for result in hp_result.all_results:
            params = result['parameters']
            score = result['mean_score']

            for param_name, param_value in params.items():
                if param_name not in param_effects:
                    param_effects[param_name] = {}
                if param_value not in param_effects[param_name]:
                    param_effects[param_name][param_value] = []
                param_effects[param_name][param_value].append(score)

        # Calculate average effects
        param_names = []
        param_ranges = []

        for param_name, values_dict in param_effects.items():
            if len(values_dict) > 1:  # Only plot parameters with multiple values
                param_names.append(param_name.replace('_', ' ').title())
                value_means = [np.mean(scores) for scores in values_dict.values()]
                param_ranges.append(max(value_means) - min(value_means))

        if param_names:
            # Create horizontal bar plot
            y_pos = np.arange(len(param_names))
            bars = ax.barh(y_pos, param_ranges, color=self.vmamba_colors['accent'], alpha=0.7)

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + max(param_ranges) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{width:.4f}', ha='left', va='center', fontsize=9)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(param_names)
            ax.set_xlabel('Score Range')
            ax.set_title('Parameter Sensitivity Analysis', fontweight='bold')

    def _plot_score_distribution(self, ax, hp_result: HyperparameterSearchResult):
        """Plot score distribution histogram"""
        if not hp_result.all_results:
            ax.text(0.5, 0.5, 'No Results Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return

        all_scores = [r['mean_score'] for r in hp_result.all_results]

        # Create histogram
        ax.hist(all_scores, bins=20, color=self.vmamba_colors['secondary'],
               alpha=0.7, edgecolor='white', linewidth=1)

        # Add best score line
        ax.axvline(hp_result.best_score, color=self.vmamba_colors['success'],
                  linestyle='--', linewidth=2, label=f'Best: {hp_result.best_score:.4f}')

        # Add mean line
        mean_score = np.mean(all_scores)
        ax.axvline(mean_score, color=self.vmamba_colors['warning'],
                  linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')

        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution', fontweight='bold')
        ax.legend()

    def _plot_model_type_comparison(self, ax, hp_result: HyperparameterSearchResult):
        """Plot model type performance comparison"""
        if not hp_result.statistical_analysis or 'model_comparison' not in hp_result.statistical_analysis:
            ax.text(0.5, 0.5, 'No Model Comparison Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return

        model_comp = hp_result.statistical_analysis['model_comparison']

        models = []
        means = []
        stds = []

        for model_type, stats in model_comp.items():
            if stats['mean'] is not None:
                models.append(model_type.replace('_', ' ').upper())
                means.append(stats['mean'])
                stds.append(stats['std'] or 0)

        if models:
            x_pos = np.arange(len(models))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                         color=[self.vmamba_colors['primary'], self.vmamba_colors['accent']][:len(models)],
                         alpha=0.7, edgecolor='white', linewidth=1)

            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + max(means) * 0.01,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(models)
            ax.set_ylabel('Mean Score')
            ax.set_title('Model Type Comparison', fontweight='bold')

    def _plot_statistical_analysis(self, ax, hp_result: HyperparameterSearchResult):
        """Plot statistical analysis results"""
        ax.axis('off')

        if not hp_result.statistical_analysis:
            ax.text(0.5, 0.5, 'No Statistical Analysis', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return

        stats = hp_result.statistical_analysis

        # Create statistical summary text
        summary = "Statistical Analysis\n\n"
        summary += f"Total Experiments: {stats.get('total_experiments', 'N/A')}\n"
        summary += f"Overall Mean: {stats.get('overall_mean', 0):.4f}\n"
        summary += f"Overall Std: {stats.get('overall_std', 0):.4f}\n\n"

        if 'statistical_test' in stats:
            test_info = stats['statistical_test']
            summary += f"Statistical Test: {test_info['method'].title()}\n"
            summary += f"P-value: {test_info.get('p_value', 0):.4f}\n"
            significance = "Yes" if test_info.get('significant', False) else "No"
            summary += f"Significant: {significance}\n"

        if getattr(hp_result, 'best_objective_scores', None):
            summary += "\nObjective Scores\n"
            for name, value in hp_result.best_objective_scores.items():
                label = name.replace('_', ' ').title()
                summary += f"{label}: {value:.4f}\n"

        if cfg.hyperparameter.objective_weights:
            summary += "\nObjective Weights\n"
            for name, weight in cfg.hyperparameter.objective_weights.items():
                label = name.replace('_', ' ').title()
                summary += f"{label}: {weight:.2f}\n"

        # Style the text box
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor=self.vmamba_colors['neutral_light'],
                         edgecolor=self.vmamba_colors['secondary'], linewidth=1, alpha=0.8)

        ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=bbox_props, family='monospace')

    def _plot_parameter_correlation(self, ax, hp_result: HyperparameterSearchResult):
        """Plot parameter correlation heatmap"""
        if not hp_result.all_results:
            ax.text(0.5, 0.5, 'No Results Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return

        param_rows: List[Dict[str, float]] = []
        numeric_keys: Set[str] = set()

        for result in hp_result.all_results:
            params = result.get('parameters', {})
            row: Dict[str, float] = {}
            for param_name, param_value in params.items():
                if isinstance(param_value, (int, float)):
                    row[param_name] = float(param_value)
                    numeric_keys.add(param_name)
            if row:
                param_rows.append(row)

        if not param_rows or len(numeric_keys) < 2:
            ax.text(0.5, 0.5, 'Insufficient Numeric Parameters', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return

        param_df = pd.DataFrame(param_rows, columns=sorted(numeric_keys))
        param_df = param_df.apply(pd.to_numeric, errors='coerce')
        param_df = param_df.dropna(axis=1, how='all')
        param_df = param_df.loc[:, param_df.apply(lambda col: col.nunique() > 1)]

        if param_df.shape[1] < 2:
            ax.text(0.5, 0.5, 'Insufficient Variation', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return

        correlation_matrix = param_df.corr().fillna(0.0)
        param_names = correlation_matrix.columns.tolist()

        im = ax.imshow(correlation_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        for i in range(len(param_names)):
            for j in range(len(param_names)):
                text_color = 'white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black'
                ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                       ha='center', va='center', color=text_color, fontsize=8)

        ax.set_xticks(range(len(param_names)))
        ax.set_yticks(range(len(param_names)))
        ax.set_xticklabels([name.replace('_', ' ') for name in param_names], rotation=45)
        ax.set_yticklabels([name.replace('_', ' ') for name in param_names])
        ax.set_title('Parameter Correlation', fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation', rotation=270, labelpad=15)

    def plot_hyperparameter_heatmap(self, hp_result: HyperparameterSearchResult,
                                   param1: str, param2: str,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 2D heatmap for two specific hyperparameters

        Args:
            hp_result: Hyperparameter search results
            param1: First parameter name
            param2: Second parameter name
            save_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        if not hp_result.all_results:
            ax.text(0.5, 0.5, 'No Results Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return fig

        # Extract parameter values and scores
        data_points = []
        for result in hp_result.all_results:
            params = result['parameters']
            if param1 in params and param2 in params:
                data_points.append({
                    'param1': params[param1],
                    'param2': params[param2],
                    'score': result['mean_score']
                })

        if not data_points:
            ax.text(0.5, 0.5, f'No data for {param1} vs {param2}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig

        # Create DataFrame and pivot for heatmap
        df = pd.DataFrame(data_points)
        heatmap_data = df.pivot_table(values='score', index='param2', columns='param1', aggfunc='mean')

        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', ax=ax,
                   fmt='.4f', cbar_kws={'label': 'Score'})

        ax.set_title(f'Hyperparameter Heatmap: {param1.replace("_", " ").title()} vs {param2.replace("_", " ").title()}',
                    fontweight='bold', fontsize=14)
        ax.set_xlabel(param1.replace('_', ' ').title())
        ax.set_ylabel(param2.replace('_', ' ').title())

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path, f"heatmap_{param1}_{param2}")

        return fig

    def create_hyperparameter_dashboard(self, hp_result: HyperparameterSearchResult,
                                      save_path: Optional[str] = None) -> str:
        """
        Create interactive dashboard for hyperparameter analysis

        Args:
            hp_result: Hyperparameter search results
            save_path: Path to save the dashboard

        Returns:
            Path to saved dashboard file
        """
        if not hp_result.all_results:
            logger.warning("No results available for dashboard creation")
            return ""

        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Search Progress', 'Score Distribution', 'Parameter Sensitivity',
                          'Model Comparison', 'Best Parameters', 'Convergence Details'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "table"}, {"type": "scatter"}]]
        )

        # 1. Search Progress
        iterations = [r['iteration'] for r in hp_result.search_history]
        scores = [r['mean_score'] for r in hp_result.search_history]

        fig.add_trace(
            go.Scatter(x=iterations, y=scores, mode='lines+markers',
                      name='Search Progress', line=dict(color=self.vmamba_colors['primary'])),
            row=1, col=1
        )

        # 2. Score Distribution
        fig.add_trace(
            go.Histogram(x=scores, name='Score Distribution',
                        marker=dict(color=self.vmamba_colors['secondary'])),
            row=1, col=2
        )

        # 3. Parameter Sensitivity (simplified)
        param_names = list(hp_result.best_params.keys())[:5]  # Top 5 parameters
        param_values = [str(hp_result.best_params[p]) for p in param_names]

        fig.add_trace(
            go.Bar(x=param_names, y=[1]*len(param_names), name='Best Parameters',
                   text=param_values, textposition='auto',
                   marker=dict(color=self.vmamba_colors['accent'])),
            row=1, col=3
        )

        # Update layout
        fig.update_layout(
            title="Hyperparameter Optimization Dashboard",
            title_font_size=16,
            showlegend=False,
            height=800
        )

        # Save dashboard
        if save_path:
            dashboard_file = os.path.join(save_path, "hyperparameter_dashboard.html")
            fig.write_html(dashboard_file)
            logger.info(f"Interactive dashboard saved: {dashboard_file}")
            return dashboard_file

        return ""
