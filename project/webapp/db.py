"""Database setup for violation records - Version 1: simple, no accounts yet."""

import sqlite3
import hashlib
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "violations.db")


def get_connection():
    """Opens a connection to our database file."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access results by column name, not just position
    return conn


def init_db():
    """Creates the violations and accounts tables if they don't exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rider_id INTEGER NOT NULL,
            violation_type TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            snapshot_path TEXT,
            status TEXT NOT NULL DEFAULT 'pending'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
            email TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_signups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            otp TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            otp TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def hash_password(password):
    """Turns a plain-text password into a scrambled hash - so we
    never store the real password anywhere."""
    return hashlib.sha256(password.encode()).hexdigest()


def create_account(username, password, role):
    conn = get_connection()
    conn.execute("""
        INSERT INTO accounts (username, password_hash, role, created_at)
        VALUES (?, ?, ?, ?)
    """, (username, hash_password(password), role, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def verify_login(username, password):
    """Checks if a username/password combination is correct.
    Returns the account dict if valid, None otherwise."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM accounts WHERE username = ?", (username,)).fetchone()
    conn.close()
    if row and row["password_hash"] == hash_password(password):
        return dict(row)
    return None


if __name__ == "__main__":
    init_db()
    print(f"Database ready at: {DB_PATH}")


def insert_violation(rider_id, violation_type, snapshot_path, confidence=None, video_source=None, uploaded_by=None, video_start_time=None, video_fps=None):
    """Adds one new violation record to the database. Returns the new row's id."""
    conn = get_connection()
    cursor = conn.execute("""
        INSERT INTO violations (rider_id, violation_type, detected_at, snapshot_path, confidence, video_source, uploaded_by, video_start_time, video_fps)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (rider_id, violation_type, datetime.now().isoformat(), snapshot_path, confidence, video_source, uploaded_by, video_start_time, video_fps))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_violations_for_video(video_source):
    """Fetches only the violations that came from one specific video."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM violations WHERE video_source = ? ORDER BY id DESC", (video_source,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_violations():
    """Fetches every violation currently in the database."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM violations ORDER BY detected_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_status(violation_id, new_status):
    """Changes a violation's status to 'approved' or 'rejected'."""
    conn = get_connection()
    conn.execute("UPDATE violations SET status = ? WHERE id = ?", (new_status, violation_id))
    conn.commit()
    conn.close()


def get_violation_by_id(violation_id):
    """Fetches a single violation by its ID."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM violations WHERE id = ?", (violation_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_pending_violations():
    """All violations still awaiting admin review, across all users."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM violations WHERE status = 'pending' ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_reviewed_violations():
    """All violations an admin has already approved or rejected."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM violations WHERE status != 'pending' ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_latest_video_for_user(username):
    """Finds the filename of the most recent video this user uploaded.
    Returns None if they've never uploaded anything."""
    conn = get_connection()
    row = conn.execute(
        "SELECT video_source FROM violations WHERE uploaded_by = ? ORDER BY id DESC LIMIT 1",
        (username,)
    ).fetchone()
    conn.close()
    return row["video_source"] if row else None


def get_violations_for_user_video(username, video_source):
    """This user's violations from one specific video of theirs."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM violations WHERE uploaded_by = ? AND video_source = ? ORDER BY id DESC",
        (username, video_source)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_past_violations_for_user(username, exclude_video_source):
    """This user's violations from any video OTHER than their current one."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM violations WHERE uploaded_by = ? AND video_source != ? ORDER BY id DESC",
        (username, exclude_video_source)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def create_pending_signup(username, email, password, otp):
    """Stores a signup attempt while we wait for OTP verification.
    The real account isn't created until the OTP is confirmed."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO pending_signups (username, email, password_hash, otp, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (username, email, hash_password(password), otp, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_pending_signup(username):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM pending_signups WHERE username = ? ORDER BY id DESC LIMIT 1", (username,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def verify_otp_and_create_account(username, submitted_otp):
    """Checks the OTP; if correct, creates the real account and
    cleans up the pending signup. Returns True if successful."""
    pending = get_pending_signup(username)
    if pending is None or pending["otp"] != submitted_otp:
        return False

    conn = get_connection()
    conn.execute("""
        INSERT INTO accounts (username, password_hash, role, email, created_at)
        VALUES (?, ?, 'user', ?, ?)
    """, (pending["username"], pending["password_hash"], pending["email"], datetime.now().isoformat()))
    conn.execute("DELETE FROM pending_signups WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    return True


def username_exists(username):
    conn = get_connection()
    row = conn.execute("SELECT id FROM accounts WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row is not None


def email_exists(email):
    conn = get_connection()
    row = conn.execute("SELECT id FROM accounts WHERE email = ?", (email,)).fetchone()
    conn.close()
    return row is not None


def get_account_by_email(email):
    conn = get_connection()
    row = conn.execute("SELECT * FROM accounts WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None


def reset_password(username, new_password):
    conn = get_connection()
    conn.execute(
        "UPDATE accounts SET password_hash = ? WHERE username = ?",
        (hash_password(new_password), username)
    )
    conn.commit()
    conn.close()


def create_password_reset(username, otp):
    conn = get_connection()
    conn.execute("""
        INSERT INTO password_resets (username, otp, created_at)
        VALUES (?, ?, ?)
    """, (username, otp, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def verify_reset_otp(username, submitted_otp):
    """Checks the OTP is correct for this username's most recent reset request."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM password_resets WHERE username = ? ORDER BY id DESC LIMIT 1", (username,)
    ).fetchone()
    conn.close()
    if row is None:
        return False
    return row["otp"] == submitted_otp


def get_stats():
    """Returns summary counts for the admin dashboard."""
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) as c FROM violations").fetchone()["c"]
    pending = conn.execute("SELECT COUNT(*) as c FROM violations WHERE status = 'pending'").fetchone()["c"]
    approved = conn.execute("SELECT COUNT(*) as c FROM violations WHERE status = 'approved'").fetchone()["c"]
    rejected = conn.execute("SELECT COUNT(*) as c FROM violations WHERE status = 'rejected'").fetchone()["c"]
    conn.close()
    return {"total": total, "pending": pending, "approved": approved, "rejected": rejected}


def get_reviewed_violations_for_user(username):
    """All of this user's violations that have been reviewed
    (approved or rejected), across any video they've uploaded."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM violations WHERE uploaded_by = ? AND status != 'pending' ORDER BY id DESC",
        (username,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_pending_past_violations_for_user(username, exclude_video_source):
    """This user's still-pending violations from videos OTHER than
    their current one."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM violations WHERE uploaded_by = ? AND video_source != ? AND status = 'pending' ORDER BY id DESC",
        (username, exclude_video_source)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_violations_by_id_range(username, min_id):
    """This user's violations with an id >= min_id - used to identify
    exactly which violations came from one specific upload session,
    since filenames alone can repeat across different uploads."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM violations WHERE uploaded_by = ? AND id >= ? ORDER BY id DESC",
        (username, min_id)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_violations_below_id(username, max_id):
    """This user's violations with id < max_id - i.e. everything
    from BEFORE their current upload session."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM violations WHERE uploaded_by = ? AND id < ? ORDER BY id DESC",
        (username, max_id)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
