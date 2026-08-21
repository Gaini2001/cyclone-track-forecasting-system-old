"""
save_model.py

Persistence for trained model artifacts.

Changes from the original
-------------------------
* The TensorFlow/Keras branch is gone. Nothing in this project produces a
  .keras or .h5 file -- the sequence-model modules exist only as orphaned .pyc
  files -- so the branch advertised a capability the repo does not have and
  carried an unused heavyweight import.
* No default destination. `save_model(model)` previously wrote to the Random
  Forest path whatever it was handed, so a mistaken call would quietly save an
  XGBoost model under a Random Forest filename.
* Writes are atomic. joblib.dump wrote straight to the target, so an
  interrupted save left a truncated file that still passed an exists() check
  and failed much later with an opaque unpickling error.
* Artifacts are compressed. A large forest can run to hundreds of megabytes
  uncompressed, which is the difference between a repo that can be pushed and
  one that cannot.
* The library versions that produced an artifact are captured, because
  unpickling an estimator under a different scikit-learn version is a known
  source of quiet misbehaviour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

from src.utils import get_logger

logger = get_logger(__name__)

# joblib compression level. 3 is the usual sweet spot: most of the size
# reduction, little of the time cost.
DEFAULT_COMPRESSION = 3

# Libraries whose version can change how a pickled estimator behaves.
TRACKED_LIBRARIES = ("numpy", "scipy", "sklearn", "xgboost", "joblib")


# ==========================================================
# Environment
# ==========================================================

def capture_environment() -> dict:
    """
    Record the versions that produced an artifact.

    Include this in the metadata written alongside a model. When a result
    cannot be reproduced months later, "which scikit-learn was this?" is
    usually the first useful question and never the one anybody wrote down.
    """

    environment = {"python": sys.version.split()[0], "platform": sys.platform}

    for name in TRACKED_LIBRARIES:
        try:
            module = __import__(name)
            environment[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue

    return environment


def check_environment(recorded: dict) -> list[str]:
    """
    Compare a recorded environment against the current one.

    Returns a list of human-readable mismatches; empty means consistent.
    """

    if not recorded:
        return []

    current = capture_environment()

    return [
        f"{name}: saved with {version}, running {current[name]}"
        for name, version in recorded.items()
        if name in current and current[name] != version
    ]


# ==========================================================
# Save
# ==========================================================

def save_model(
    model,
    model_path: Path | str,
    compress: int = DEFAULT_COMPRESSION,
    metadata: dict = None,
) -> Path:
    """
    Persist a trained estimator.

    Parameters
    ----------
    model : object
        Any joblib-serializable estimator.
    model_path : Path or str
        Destination. Required -- there is no sensible default, and a default
        here means a wrong call writes a correctly-named wrong file.
    compress : int
        joblib compression level, 0-9. 0 disables compression.
    metadata : dict, optional
        Written to a JSON sidecar beside the artifact, merged with any existing
        sidecar and with the current environment added.

    Returns
    -------
    Path
        The written path.
    """

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file and rename on success, so an interrupted save
    # cannot leave a truncated artifact that later looks valid.
    temp_path = model_path.with_suffix(model_path.suffix + ".tmp")

    logger.info(f"Saving model to: {model_path}")

    try:
        joblib.dump(model, temp_path, compress=compress)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    temp_path.replace(model_path)

    size_mb = model_path.stat().st_size / 1024 ** 2
    logger.info(f"  Saved ({size_mb:.1f} MB, compress={compress})")

    if metadata is not None:
        write_metadata(model_path, metadata)

    return model_path


def write_metadata(model_path: Path | str, metadata: dict) -> Path:
    """
    Write or update the JSON sidecar for an artifact.

    Merges rather than overwrites, so a caller that adds training details does
    not erase the environment record, and vice versa.
    """

    model_path = Path(model_path)
    metadata_path = model_path.with_suffix(".json")

    existing = {}

    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            logger.warning(f"  Replacing unreadable sidecar: {metadata_path.name}")

    merged = {**existing, **metadata}
    merged.setdefault("environment", capture_environment())

    metadata_path.write_text(json.dumps(merged, indent=2, default=str))

    return metadata_path


# ==========================================================
# Load
# ==========================================================

def load_model(model_path: Path | str, warn_on_version_mismatch: bool = True):
    """
    Load a trained estimator.

    Parameters
    ----------
    warn_on_version_mismatch : bool
        Compare the recorded environment against the current one and log any
        differences. Not fatal -- most version changes are harmless -- but the
        warning is what makes an otherwise inexplicable behaviour change
        traceable.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"No model at {model_path}. Train one first:\n"
            f"  python -m src.s5_training.train_model --model all --horizon all"
        )

    logger.info(f"Loading model: {model_path.name}")

    model = joblib.load(model_path)

    if warn_on_version_mismatch:
        metadata = read_metadata(model_path)
        mismatches = check_environment(metadata.get("environment", {}))

        if mismatches:
            logger.warning(
                f"  {model_path.name} was saved under a different environment:"
            )
            for mismatch in mismatches:
                logger.warning(f"    {mismatch}")

    return model


def read_metadata(model_path: Path | str) -> dict:
    """
    Read an artifact's sidecar, or an empty dict if there isn't one.
    """

    metadata_path = Path(model_path).with_suffix(".json")

    if not metadata_path.exists():
        return {}

    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Unreadable sidecar: {metadata_path}")
        return {}


def load_model_with_metadata(model_path: Path | str) -> tuple:
    """
    Load an estimator together with its sidecar.

    Returns
    -------
    (model, metadata) : tuple
    """

    return load_model(model_path), read_metadata(model_path)


# ==========================================================
# Inventory
# ==========================================================

def list_models(model_dir: Path | str = None) -> list[dict]:
    """
    Summarize the artifacts on disk.

    Useful before evaluating or predicting: it answers "what have I actually
    trained?" without opening five files.
    """

    from src.utils.config import MODEL_DIR

    model_dir = Path(model_dir) if model_dir else MODEL_DIR

    if not model_dir.exists():
        return []

    inventory = []

    for path in sorted(model_dir.glob("*.pkl")):
        metadata = read_metadata(path)

        inventory.append({
            "file": path.name,
            "size_mb": round(path.stat().st_size / 1024 ** 2, 1),
            "model": metadata.get("model"),
            "horizon": metadata.get("horizon"),
            "target_mode": metadata.get("target_mode"),
            "n_features": metadata.get("n_features"),
            "trained_at": metadata.get("trained_at"),
        })

    return inventory


def log_model_inventory(model_dir: Path | str = None) -> None:
    """
    Print the artifact inventory.
    """

    inventory = list_models(model_dir)

    if not inventory:
        logger.info("No trained models found.")
        return

    logger.info("=" * 76)
    logger.info("TRAINED MODELS")
    logger.info("=" * 76)
    logger.info(
        f"  {'File':<44} {'Horizon':>8} {'Feats':>6} {'Size':>8}"
    )
    logger.info("  " + "-" * 70)

    for entry in inventory:
        horizon = f"{entry['horizon']}h" if entry["horizon"] else "?"
        logger.info(
            f"  {entry['file']:<44} {horizon:>8} "
            f"{str(entry['n_features'] or '?'):>6} {entry['size_mb']:>7.1f}M"
        )

    logger.info("=" * 76)


if __name__ == "__main__":
    log_model_inventory()