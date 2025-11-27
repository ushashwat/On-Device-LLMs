"""logger_config.py script for configuring script-specific logger."""
import logging
from pathlib import Path

def setup_logger(logger_name, log_file, level=logging.INFO):
    """Sets up a logger to save logs to a specific directory."""
    # Get the project root to build the correct path for logs
    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / log_file

    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Use FileHandler to log messages to a file
    handler = logging.FileHandler(log_file_path, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(handler)

    # Stop logs from propagating to the root logger
    logger.propagate = False

    return logger
