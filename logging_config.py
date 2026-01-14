import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_file: str = "app.log",
) -> None:
    """
    Configure a console + rotating file logger.
    Idempotent: updates existing handlers if they already exist.
    """
    os.makedirs(log_dir, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # If handlers exist, update them instead of adding new ones
    if logger.handlers:
        for h in logger.handlers:
            h.setLevel(level)
            h.setFormatter(formatter)
        return

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating file
    fh = RotatingFileHandler(
        filename=os.path.join(log_dir, log_file),
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)