"""
Logging configuration for PyART

This module provides a simple logging configuration that can be used throughout PyART.
Users can call setup_logging() to configure logging for the entire package.

Example usage:
    import PyART
    from PyART.logging_config import setup_logging
    
    # Setup logging with default INFO level
    setup_logging()
    
    # Or setup with custom level
    setup_logging(level='DEBUG')
"""

import logging


def setup_logging(level='INFO', format_string=None, datefmt=None):
    """
    Setup logging configuration for PyART
    
    Parameters
    ----------
    level : str, optional
        Logging level. Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
        Default is 'INFO'.
    format_string : str, optional
        Custom format string for log messages.
        Default is '%(asctime)s %(message)s'
    datefmt : str, optional
        Date format for timestamps in log messages.
        Default is '%Y-%m-%d %H:%M:%S'
    
    Example
    -------
    >>> from PyART.logging_config import setup_logging
    >>> setup_logging(level='INFO')
    >>> import logging
    >>> logging.info("This is an info message")
    2024-01-01 12:00:00 This is an info message
    """
    if format_string is None:
        format_string = '%(asctime)s %(message)s'
    if datefmt is None:
        datefmt = '%Y-%m-%d %H:%M:%S'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt=datefmt
    )
