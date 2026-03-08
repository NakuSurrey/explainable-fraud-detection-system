"""
Phase 10: Feedback Manager -- Human-in-the-Loop Database Module
================================================================
Manages investigator feedback on fraud predictions using SQLite.

Functions:
    generate_transaction_id(features_dict) -- SHA-256 hash of transaction features
    init_db() -- Create feedback table if not exists
    save_feedback(record) -- Insert a new feedback record
    get_feedback_history(limit, offset) -- Query feedback with pagination
    get_feedback_by_id(feedback_id) -- Get a single feedback record
    get_feedback_by_transaction(transaction_id) -- Get feedback for a transaction
    update_feedback(feedback_id, updates) -- Update an existing record
    delete_feedback(feedback_id) -- Delete a feedback record
    export_corrections(output_path) -- Export all corrections to CSV
    check_retrain_threshold() -- Check if enough corrections for retraining
    get_feedback_stats() -- Summary statistics of all feedback

All paths come from config.yaml. All logging goes through centralized logger.
No print() statements. ASCII-only log messages.
"""

import sqlite3
import hashlib
import json
import csv
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from src.utils.logger import load_config, get_logger, resolve_path, log_phase_start, log_phase_end

# ---------------------------------------------------------------------------
# Module setup
# ---------------------------------------------------------------------------
logger = get_logger("feedback.feedback_manager")
PHASE_NAME = "Phase 10: Feedback Manager"


# ---------------------------------------------------------------------------
# 1. TRANSACTION ID GENERATION
# ---------------------------------------------------------------------------

def generate_transaction_id(features_dict: dict) -> str:
    """
    Generate a deterministic transaction ID from feature values.

    Uses SHA-256 hash of sorted feature key-value pairs so the same
    transaction always produces the same ID regardless of dict ordering.

    Args:
        features_dict: Dictionary of feature names to values
                       (e.g., {"V1": -1.35, "V2": 1.19, ..., "Amount": 149.62})

    Returns:
        A 16-character hex string (first 16 chars of SHA-256 hash).
        Example: "a3f7b2c1d4e5f6a7"
    """
    # Sort keys for deterministic ordering
    sorted_items = sorted(features_dict.items(), key=lambda x: x[0])
    # Create a stable string representation
    feature_string = "|".join(f"{k}:{v}" for k, v in sorted_items)
    # Hash it
    full_hash = hashlib.sha256(feature_string.encode("utf-8")).hexdigest()
    # Return first 16 characters (64 bits -- collision-resistant for our scale)
    return full_hash[:16]


# ---------------------------------------------------------------------------
# 2. DATABASE INITIALIZATION
# ---------------------------------------------------------------------------

def _get_db_path(config: Optional[dict] = None) -> Path:
    """Resolve the feedback database path from config."""
    if config is None:
        config = load_config()
    db_path = resolve_path(config["feedback"]["db_path"])
    return db_path


def _get_connection(config: Optional[dict] = None) -> sqlite3.Connection:
    """
    Get a SQLite connection with row_factory set for dict-like access.

    Creates the parent directory if it doesn't exist.
    """
    db_path = _get_db_path(config)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrent read performance
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(config: Optional[dict] = None) -> str:
    """
    Initialize the feedback database schema.

    Creates the feedback table if it doesn't already exist.
    Safe to call multiple times (idempotent).

    Returns:
        The absolute path to the database file as a string.
    """
    conn = _get_connection(config)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id  TEXT NOT NULL,
                original_probability    REAL NOT NULL,
                original_risk_level     TEXT NOT NULL,
                original_is_flagged     INTEGER NOT NULL,
                correction_type         TEXT NOT NULL CHECK(correction_type IN ('confirmed_fraud', 'false_positive')),
                investigator_notes      TEXT DEFAULT '',
                created_at              TEXT NOT NULL,
                updated_at              TEXT
            )
        """)
        # Index on transaction_id for fast lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_transaction_id
            ON feedback(transaction_id)
        """)
        # Index on correction_type for filtering
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_correction_type
            ON feedback(correction_type)
        """)
        conn.commit()
        db_path = _get_db_path(config)
        logger.info(f"Feedback database initialized at: {db_path}")
        return str(db_path)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3. SAVE FEEDBACK
# ---------------------------------------------------------------------------

def save_feedback(
    transaction_id: str,
    original_probability: float,
    original_risk_level: str,
    original_is_flagged: bool,
    correction_type: str,
    investigator_notes: str = "",
    config: Optional[dict] = None
) -> dict:
    """
    Save a new feedback record to the database.

    Args:
        transaction_id: Hash-based ID from generate_transaction_id()
        original_probability: Model's fraud probability (0.0 to 1.0)
        original_risk_level: Model's risk classification (LOW/MEDIUM/HIGH/CRITICAL)
        original_is_flagged: Whether model flagged as fraudulent (True/False)
        correction_type: Either 'confirmed_fraud' or 'false_positive'
        investigator_notes: Optional free-text notes from the investigator
        config: Optional config dict (loads from file if not provided)

    Returns:
        Dict with the saved record details including the new id.

    Raises:
        ValueError: If correction_type is not valid.
    """
    # Validate correction_type
    valid_types = ("confirmed_fraud", "false_positive")
    if correction_type not in valid_types:
        raise ValueError(
            f"Invalid correction_type: '{correction_type}'. "
            f"Must be one of: {valid_types}"
        )

    created_at = datetime.now(timezone.utc).isoformat()

    conn = _get_connection(config)
    try:
        cursor = conn.execute(
            """
            INSERT INTO feedback (
                transaction_id, original_probability, original_risk_level,
                original_is_flagged, correction_type, investigator_notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                original_probability,
                original_risk_level,
                1 if original_is_flagged else 0,
                correction_type,
                investigator_notes,
                created_at,
            ),
        )
        conn.commit()
        record_id = cursor.lastrowid

        logger.info(
            f"Feedback saved -- id={record_id}, "
            f"txn={transaction_id[:8]}..., "
            f"correction={correction_type}, "
            f"original_prob={original_probability:.4f}"
        )

        return {
            "id": record_id,
            "transaction_id": transaction_id,
            "original_probability": original_probability,
            "original_risk_level": original_risk_level,
            "original_is_flagged": bool(original_is_flagged),
            "correction_type": correction_type,
            "investigator_notes": investigator_notes,
            "created_at": created_at,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 4. QUERY FEEDBACK
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dict with proper types."""
    d = dict(row)
    # Convert integer flag back to boolean
    d["original_is_flagged"] = bool(d["original_is_flagged"])
    return d


def get_feedback_history(
    limit: int = 50,
    offset: int = 0,
    correction_type: Optional[str] = None,
    config: Optional[dict] = None
) -> dict:
    """
    Query feedback history with pagination and optional filtering.

    Args:
        limit: Maximum number of records to return (default 50)
        offset: Number of records to skip (for pagination)
        correction_type: Optional filter -- 'confirmed_fraud' or 'false_positive'
        config: Optional config dict

    Returns:
        Dict with 'records' (list of dicts), 'total' (total count),
        'limit', and 'offset'.
    """
    conn = _get_connection(config)
    try:
        # Build query with optional filter
        where_clause = ""
        params = []
        if correction_type:
            where_clause = "WHERE correction_type = ?"
            params.append(correction_type)

        # Get total count
        count_query = f"SELECT COUNT(*) FROM feedback {where_clause}"
        total = conn.execute(count_query, params).fetchone()[0]

        # Get paginated records (newest first)
        data_query = f"""
            SELECT * FROM feedback {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        data_params = params + [limit, offset]
        rows = conn.execute(data_query, data_params).fetchall()

        records = [_row_to_dict(row) for row in rows]

        return {
            "records": records,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    finally:
        conn.close()


def get_feedback_by_id(feedback_id: int, config: Optional[dict] = None) -> Optional[dict]:
    """
    Get a single feedback record by its database ID.

    Args:
        feedback_id: The auto-increment ID of the record.
        config: Optional config dict.

    Returns:
        Dict of the record, or None if not found.
    """
    conn = _get_connection(config)
    try:
        row = conn.execute(
            "SELECT * FROM feedback WHERE id = ?", (feedback_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)
    finally:
        conn.close()


def get_feedback_by_transaction(
    transaction_id: str, config: Optional[dict] = None
) -> list:
    """
    Get all feedback records for a specific transaction.

    A transaction can have multiple feedback entries (e.g., if an
    investigator changes their mind).

    Args:
        transaction_id: The hash-based transaction ID.
        config: Optional config dict.

    Returns:
        List of record dicts (may be empty).
    """
    conn = _get_connection(config)
    try:
        rows = conn.execute(
            "SELECT * FROM feedback WHERE transaction_id = ? ORDER BY created_at DESC",
            (transaction_id,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 5. UPDATE FEEDBACK
# ---------------------------------------------------------------------------

def update_feedback(
    feedback_id: int,
    correction_type: Optional[str] = None,
    investigator_notes: Optional[str] = None,
    config: Optional[dict] = None
) -> Optional[dict]:
    """
    Update an existing feedback record.

    Only correction_type and investigator_notes can be updated.
    The original prediction data is immutable (audit trail).

    Args:
        feedback_id: The ID of the record to update.
        correction_type: New correction type (optional).
        investigator_notes: New notes (optional).
        config: Optional config dict.

    Returns:
        Updated record dict, or None if record not found.

    Raises:
        ValueError: If no update fields provided or invalid correction_type.
    """
    if correction_type is None and investigator_notes is None:
        raise ValueError("At least one of correction_type or investigator_notes must be provided")

    if correction_type is not None:
        valid_types = ("confirmed_fraud", "false_positive")
        if correction_type not in valid_types:
            raise ValueError(
                f"Invalid correction_type: '{correction_type}'. "
                f"Must be one of: {valid_types}"
            )

    # Check record exists
    existing = get_feedback_by_id(feedback_id, config)
    if existing is None:
        logger.warning(f"Feedback update failed -- record id={feedback_id} not found")
        return None

    # Build dynamic UPDATE
    updates = []
    params = []
    if correction_type is not None:
        updates.append("correction_type = ?")
        params.append(correction_type)
    if investigator_notes is not None:
        updates.append("investigator_notes = ?")
        params.append(investigator_notes)

    updates.append("updated_at = ?")
    updated_at = datetime.now(timezone.utc).isoformat()
    params.append(updated_at)

    params.append(feedback_id)

    conn = _get_connection(config)
    try:
        conn.execute(
            f"UPDATE feedback SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()
        logger.info(f"Feedback updated -- id={feedback_id}")
    finally:
        conn.close()

    # Return the updated record
    return get_feedback_by_id(feedback_id, config)


# ---------------------------------------------------------------------------
# 6. DELETE FEEDBACK
# ---------------------------------------------------------------------------

def delete_feedback(feedback_id: int, config: Optional[dict] = None) -> bool:
    """
    Delete a feedback record by ID.

    Args:
        feedback_id: The ID of the record to delete.
        config: Optional config dict.

    Returns:
        True if a record was deleted, False if not found.
    """
    conn = _get_connection(config)
    try:
        cursor = conn.execute(
            "DELETE FROM feedback WHERE id = ?", (feedback_id,)
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Feedback deleted -- id={feedback_id}")
        else:
            logger.warning(f"Feedback delete failed -- id={feedback_id} not found")
        return deleted
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 7. EXPORT CORRECTIONS TO CSV
# ---------------------------------------------------------------------------

def export_corrections(output_path: Optional[str] = None, config: Optional[dict] = None) -> dict:
    """
    Export all feedback corrections to a CSV file.

    This is a manual operation -- only runs when explicitly called
    (e.g., via the /feedback/export API endpoint).

    Args:
        output_path: Override path for the CSV. If None, uses config value.
        config: Optional config dict.

    Returns:
        Dict with 'path' (absolute path to CSV), 'total_records' (count),
        and 'exported_at' (timestamp).
    """
    if config is None:
        config = load_config()

    if output_path is None:
        csv_path = resolve_path(config["feedback"]["export_path"])
    else:
        csv_path = Path(output_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    conn = _get_connection(config)
    try:
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY created_at ASC"
        ).fetchall()

        total = len(rows)

        if total == 0:
            logger.info("Export skipped -- no feedback records to export")
            return {
                "path": str(csv_path),
                "total_records": 0,
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }

        # Write CSV
        fieldnames = [
            "id", "transaction_id", "original_probability",
            "original_risk_level", "original_is_flagged",
            "correction_type", "investigator_notes",
            "created_at", "updated_at",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                record = dict(row)
                record["original_is_flagged"] = bool(record["original_is_flagged"])
                writer.writerow(record)

        exported_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            f"Corrections exported -- {total} records -> {csv_path}"
        )

        return {
            "path": str(csv_path),
            "total_records": total,
            "exported_at": exported_at,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 8. RETRAIN THRESHOLD CHECK
# ---------------------------------------------------------------------------

def check_retrain_threshold(config: Optional[dict] = None) -> dict:
    """
    Check if enough feedback corrections have been collected to
    justify retraining the model.

    The threshold is defined in config.yaml under feedback.retrain_threshold
    (default: 100).

    Returns:
        Dict with:
            'total_corrections': int -- number of feedback records
            'threshold': int -- retrain threshold from config
            'retrain_recommended': bool -- True if total >= threshold
            'remaining': int -- how many more corrections needed (0 if met)
    """
    if config is None:
        config = load_config()

    threshold = config["feedback"].get("retrain_threshold", 100)

    conn = _get_connection(config)
    try:
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    finally:
        conn.close()

    retrain_recommended = total >= threshold
    remaining = max(0, threshold - total)

    if retrain_recommended:
        logger.info(
            f"Retrain threshold REACHED -- {total}/{threshold} corrections. "
            f"Consider re-running Phase 3 -> Phase 5 for Model v2."
        )
    else:
        logger.info(
            f"Retrain threshold check -- {total}/{threshold} corrections "
            f"({remaining} more needed)"
        )

    return {
        "total_corrections": total,
        "threshold": threshold,
        "retrain_recommended": retrain_recommended,
        "remaining": remaining,
    }


# ---------------------------------------------------------------------------
# 9. FEEDBACK STATISTICS
# ---------------------------------------------------------------------------

def get_feedback_stats(config: Optional[dict] = None) -> dict:
    """
    Get summary statistics of all feedback in the database.

    Returns:
        Dict with total count, breakdown by correction_type,
        average original probability for each type, and date range.
    """
    conn = _get_connection(config)
    try:
        # Total count
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

        if total == 0:
            return {
                "total": 0,
                "confirmed_fraud": 0,
                "false_positive": 0,
                "avg_probability_confirmed_fraud": None,
                "avg_probability_false_positive": None,
                "oldest_feedback": None,
                "newest_feedback": None,
            }

        # Count by type
        confirmed = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE correction_type = 'confirmed_fraud'"
        ).fetchone()[0]
        false_pos = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE correction_type = 'false_positive'"
        ).fetchone()[0]

        # Average probabilities by type
        avg_prob_confirmed = conn.execute(
            "SELECT AVG(original_probability) FROM feedback WHERE correction_type = 'confirmed_fraud'"
        ).fetchone()[0]
        avg_prob_false_pos = conn.execute(
            "SELECT AVG(original_probability) FROM feedback WHERE correction_type = 'false_positive'"
        ).fetchone()[0]

        # Date range
        oldest = conn.execute(
            "SELECT MIN(created_at) FROM feedback"
        ).fetchone()[0]
        newest = conn.execute(
            "SELECT MAX(created_at) FROM feedback"
        ).fetchone()[0]

        stats = {
            "total": total,
            "confirmed_fraud": confirmed,
            "false_positive": false_pos,
            "avg_probability_confirmed_fraud": round(avg_prob_confirmed, 4) if avg_prob_confirmed else None,
            "avg_probability_false_positive": round(avg_prob_false_pos, 4) if avg_prob_false_pos else None,
            "oldest_feedback": oldest,
            "newest_feedback": newest,
        }

        logger.info(
            f"Feedback stats -- total={total}, "
            f"confirmed_fraud={confirmed}, false_positive={false_pos}"
        )

        return stats
    finally:
        conn.close()
