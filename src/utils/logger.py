"""
Centralized Logger & Config Loader
===================================
EVERY phase imports this. It gives you:
  1. load_config()       -> reads config.yaml once, caches it
  2. get_logger(name)    -> returns a named logger writing to pipeline.log
  3. log_phase_start()   -> marks a phase as RUNNING in phase_status.json
  4. log_phase_end()     -> marks a phase as SUCCESS or FAILED
  5. check_phase_completed() -> lets a phase verify its prerequisites ran

Usage:
    from src.utils.logger import get_logger, load_config, log_phase_start, log_phase_end
    config = load_config()
    logger = get_logger(__name__)
"""

import os
import sys
import yaml
import logging
import json
from datetime import datetime
from pathlib import Path


# ── Project Root Detection ──────────────────────────────────────────────────

def get_project_root() -> Path:
    """Walk up from this file until we find config.yaml."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "config.yaml").exists():
            return parent
    return Path.cwd()

PROJECT_ROOT = get_project_root()


# ── Config Loader (cached) ─────────────────────────────────────────────────

_config_cache = None

def load_config() -> dict:
    """Load config.yaml. Cached after first call."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        _config_cache = yaml.safe_load(f)
    return _config_cache

def resolve_path(relative_path: str) -> Path:
    """Turn a config path like 'data/raw/x.csv' into an absolute path."""
    return PROJECT_ROOT / relative_path


# ── Centralized Logger ─────────────────────────────────────────────────────

_logger_initialized = False

def _setup_logging():
    """Initialize file + console logging. Runs once."""
    global _logger_initialized
    if _logger_initialized:
        return

    config = load_config()
    env = config.get("environment", {})
    log_level = env.get("log_level", "INFO")
    log_file = env.get("log_file", "logs/pipeline.log")

    log_path = resolve_path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    ch.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)

    _logger_initialized = True

def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Auto-initializes on first call."""
    _setup_logging()
    return logging.getLogger(name)


# ── Phase Status Tracking ──────────────────────────────────────────────────

_status_path = PROJECT_ROOT / "logs" / "phase_status.json"

def _load_status() -> dict:
    if _status_path.exists():
        with open(_status_path, "r") as f:
            return json.load(f)
    return {}

def _save_status(data: dict):
    _status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_status_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

def log_phase_start(phase_name: str):
    logger = get_logger("PhaseTracker")
    logger.info("=" * 60)
    logger.info(f"STARTING: {phase_name}")
    logger.info("=" * 60)
    s = _load_status()
    s[phase_name] = {
        "status": "RUNNING",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
    }
    _save_status(s)

def log_phase_end(phase_name: str, status: str = "SUCCESS", error: str = None):
    logger = get_logger("PhaseTracker")
    if status == "SUCCESS":
        # logger.info(f"COMPLETED: {phase_name} ✓")
        logger.info(f"COMPLETED: {phase_name} [OK]")
    else:
        # logger.error(f"FAILED: {phase_name} — {error}")
        logger.error(f"FAILED: {phase_name} -- {error}")
    logger.info("=" * 60)
    s = _load_status()
    if phase_name in s:
        s[phase_name]["status"] = status
        s[phase_name]["completed_at"] = datetime.now().isoformat()
        s[phase_name]["error"] = error
    _save_status(s)

def check_phase_completed(phase_name: str) -> bool:
    """Check if a prerequisite phase finished successfully."""
    return _load_status().get(phase_name, {}).get("status") == "SUCCESS"

def get_all_phase_status() -> dict:
    return _load_status()
