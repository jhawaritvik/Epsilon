"""
Centralized Logging Configuration

Provides consistent logging format across all components of the
Autonomous Research Engine.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO, force: bool = True):
    """
    Configures the root logger with a consistent format.
    
    Args:
        level: Logging level (default: INFO)
        force: If True, removes existing handlers (Python 3.8+)
    """
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Remove existing handlers if force is True
    if force:
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
