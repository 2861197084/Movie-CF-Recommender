"""
Academic-standard logging utilities for MovieLens CF experiments
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional

class AcademicLogger:
    """Logger designed for academic research and experiments"""

    def __init__(self, name: str, log_dir: str = "./logs", level: int = logging.INFO):
        self.name = name
        self.log_dir = log_dir
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        simple_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )

        # File handler for detailed logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)

        # Console handler for user-friendly output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)

        self.start_time = time.time()

    def info(self, message: str, *args, **kwargs):
        """Log info message (supports printf-style args)."""
        self.logger.info(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        """Log debug message (supports printf-style args)."""
        self.logger.debug(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log warning message (supports printf-style args)."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log error message (supports printf-style args)."""
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log critical message (supports printf-style args)."""
        self.logger.critical(message, *args, **kwargs)

    def log_experiment_start(self, config: dict):
        """Log experiment start with configuration"""
        self.info("=" * 80)
        self.info("EXPERIMENT STARTED")
        self.info("=" * 80)
        self.info(f"Experiment: {config.get('experiment', {}).get('experiment_name', 'Unknown')}")
        self.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("-" * 80)
        self.info("Configuration:")
        for section, params in config.items():
            self.info(f"  {section.upper()}:")
            for key, value in params.items():
                self.info(f"    {key}: {value}")
        self.info("=" * 80)

    def log_experiment_end(self):
        """Log experiment end with duration"""
        duration = time.time() - self.start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)

        self.info("=" * 80)
        self.info("EXPERIMENT COMPLETED")
        self.info(f"Total Duration: {int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}")
        self.info("=" * 80)

    def log_phase(self, phase_name: str):
        """Log start of a major phase"""
        self.info("-" * 60)
        self.info(f"PHASE: {phase_name.upper()}")
        self.info("-" * 60)

    def log_metrics(self, metrics: dict, phase: str = ""):
        """Log evaluation metrics"""
        if phase:
            self.info(f"Metrics for {phase}:")
        else:
            self.info("Evaluation Metrics:")

        for metric_name, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {metric_name}: {value:.6f}")
            else:
                self.info(f"  {metric_name}: {value}")

    def log_progress(self, current: int, total: int, prefix: str = "Progress"):
        """Log progress for long-running operations"""
        percentage = (current / total) * 100
        self.info(f"{prefix}: {current}/{total} ({percentage:.1f}%)")

def get_logger(name: str, log_dir: str = "./logs") -> AcademicLogger:
    """Get or create a logger instance"""
    return AcademicLogger(name, log_dir)