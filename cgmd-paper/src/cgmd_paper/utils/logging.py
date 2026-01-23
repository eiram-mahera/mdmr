# src/utils/logging.py
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logger(
    name: str = "mrmd",
    log_dir: str = "./logs",
    log_file: str | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    capture_warnings: bool = True,
    max_bytes: int = 10_000_000,  # 10 MB per log file
    backup_count: int = 3,        # keep 3 rotated files
) -> logging.Logger:
    """
    Configure a logger that logs to both console and file.
    Also (optionally) routes Python warnings and uncaught exceptions into the log.

    - console_level: what goes to the screen
    - file_level: what goes to the file (usually DEBUG)
    - capture_warnings=True routes `warnings.warn(...)` into logging at WARNING level
    """
    os.makedirs(log_dir, exist_ok=True)
    if log_file is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(log_dir, f"{name}_{ts}.log")
    else:
        log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))  # overall threshold

    # Avoid duplicate handlers if called twice
    if not logger.handlers:
        # --- console ---
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(console_level)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(ch)

        # --- file (rotating) ---
        fh = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
        fh.setLevel(file_level)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(fh)

        # Make sure messages from child loggers (same hierarchy) bubble up
        logger.propagate = False

        # Route Python warnings -> logging
        if capture_warnings:
            logging.captureWarnings(True)
            # Optionally, make the warnings logger inherit our file handler/console:
            wlog = logging.getLogger("py.warnings")
            wlog.setLevel(logging.WARNING)
            if not wlog.handlers:
                wlog.addHandler(ch)
                wlog.addHandler(fh)

        # Log uncaught exceptions
        def _excepthook(exc_type, exc, tb):
            logger.exception("Uncaught exception", exc_info=(exc_type, exc, tb))
            # also print default hook behavior
            sys.__excepthook__(exc_type, exc, tb)
        sys.excepthook = _excepthook

    return logger
