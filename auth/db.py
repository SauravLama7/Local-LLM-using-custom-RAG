import sqlite3
import hashlib
import os

DB_PATH = "auth/users.db"

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    """Create tables and insert default users if not exists."""
    conn = get_connection()
    c    = conn.cursor()

    # Create users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    UNIQUE NOT NULL,
            password TEXT    NOT NULL,
            name     TEXT    NOT NULL,
            role     TEXT    NOT NULL DEFAULT 'guest'
        )
    """)

    # Create roles table
    c.execute("""
        CREATE TABLE IF NOT EXISTS role_permissions (
            role     TEXT NOT NULL,
            keyword  TEXT NOT NULL,
            PRIMARY KEY (role, keyword)
        )
    """)

    conn.commit()

    # Create audit log table
    c.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT    NOT NULL,
            username   TEXT    NOT NULL,
            role       TEXT    NOT NULL,
            query      TEXT    NOT NULL,
            tool_used  TEXT    NOT NULL,
            grounded   INTEGER NOT NULL,
            score      REAL    NOT NULL,
            sources    TEXT    NOT NULL,
            model      TEXT    NOT NULL
        )
    """)
    conn.commit()


    # Insert default users if table is empty
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        default_users = [
            ("admin", hash_password("admin123"), "Administrator", "admin"),
            ("bipin",  hash_password("bipin123"),  "Bipin KC",       "hr"),
            ("rohan",  hash_password("rohan123"),  "Rohan Sharma",   "dev"),
        ]
        c.executemany(
            "INSERT INTO users (username, password, name, role) VALUES (?,?,?,?)",
            default_users
        )

    # Insert default role permissions if empty
    c.execute("SELECT COUNT(*) FROM role_permissions")
    if c.fetchone()[0] == 0:
        default_permissions = [
            ("hr",      "HR"),
            ("hr",      "Employee"),
            ("hr",      "Policy"),
            ("hr",      "Leave"),
            ("dev",     "Technical"),
            ("dev",     "Engineering"),
            ("dev",     "Dev"),
            ("finance", "Budget"),
            ("finance", "Finance"),
        ]
        c.executemany(
            "INSERT INTO role_permissions (role, keyword) VALUES (?,?)",
            default_permissions
        )

    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username: str, password: str) -> dict | None:
    """Returns user dict if credentials valid, None otherwise."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute(
        "SELECT username, name, role FROM users WHERE username=? AND password=?",
        (username.lower(), hash_password(password))
    )
    row = c.fetchone()
    conn.close()
    if row:
        return {"username": row[0], "name": row[1], "role": row[2]}
    return None

def get_allowed_docs(role: str, available_files: list[str]) -> list[str]:
    """Filter documents based on role permissions from DB."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("SELECT keyword FROM role_permissions WHERE role=?", (role,))
    keywords = [row[0] for row in c.fetchall()]
    conn.close()

    # Admin and guest see all docs
    if not keywords:
        return available_files

    # Filter by keyword match in filename
    return [
        f for f in available_files
        if any(kw.lower() in f.lower() for kw in keywords)
    ]

def add_user(username: str, password: str, name: str, role: str) -> bool:
    """Add a new user. Returns True if successful, False if username exists."""
    try:
        conn = get_connection()
        c    = conn.cursor()
        c.execute(
            "INSERT INTO users (username, password, name, role) VALUES (?,?,?,?)",
            (username.lower(), hash_password(password), name, role)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False  # username already exists

def delete_user(username: str) -> bool:
    """Delete a user by username."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("DELETE FROM users WHERE username=?", (username.lower(),))
    affected = c.rowcount
    conn.commit()
    conn.close()
    return affected > 0

def get_all_users() -> list[dict]:
    """Get all users for admin panel."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("SELECT username, name, role FROM users")
    rows = c.fetchall()
    conn.close()
    return [{"username": r[0], "name": r[1], "role": r[2]} for r in rows]

def update_password(username: str, new_password: str) -> bool:
    """Update user password."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute(
        "UPDATE users SET password=? WHERE username=?",
        (hash_password(new_password), username.lower())
    )
    affected = c.rowcount
    conn.commit()
    conn.close()
    return affected > 0

def add_role_permission(role: str, keyword: str) -> bool:
    """Add a keyword permission for a role."""
    try:
        conn = get_connection()
        c    = conn.cursor()
        c.execute(
            "INSERT INTO role_permissions (role, keyword) VALUES (?,?)",
            (role, keyword)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def get_role_permissions(role: str) -> list[str]:
    """Get all keywords for a role."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("SELECT keyword FROM role_permissions WHERE role=?", (role,))
    keywords = [row[0] for row in c.fetchall()]
    conn.close()
    return keywords
    
def log_query(username: str, role: str, query: str, tool_used: str,
              grounded: bool, score: float, sources: list, model: str):
    """Log a query to the audit log."""
    from datetime import datetime
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        INSERT INTO audit_log (timestamp, username, role, query, tool_used, grounded, score, sources, model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        username,
        role,
        query,
        tool_used,
        int(grounded),
        score,
        ", ".join(sources),
        model
    ))
    conn.commit()
    conn.close()

def get_audit_log(limit: int = 50) -> list[dict]:
    """Get recent audit log entries."""
    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        SELECT timestamp, username, role, query, tool_used, grounded, score, sources, model
        FROM audit_log
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return [
        {
            "timestamp": r[0], "username": r[1], "role":     r[2],
            "query":     r[3], "tool_used": r[4], "grounded": bool(r[5]),
            "score":     r[6], "sources":   r[7], "model":    r[8]
        }
        for r in rows
    ]