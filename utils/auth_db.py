"""
User Authentication Database helper for Virtual Brain Inference.
Uses SQLite for storage and Werkzeug security helpers for password hashing.
"""
import sqlite3
import os
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
from config import get_config

config = get_config(os.environ.get('FLASK_ENV', 'development'))
DB_PATH = config.DATA_DIR / 'users.db'

def get_db_connection():
    """Establishes connection to the SQLite database with Row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates the users table if it doesn't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username, email, password):
    """
    Registers a new user in the database.
    
    Args:
        username: Desired username
        email: Email address
        password: Plain text password
        
    Returns:
        tuple (success_boolean, user_id_or_error_message)
    """
    if not username or not email or not password:
        return False, "All fields are required."
    
    username = username.strip()
    email = email.strip().lower()
    
    # Simple length checks
    if len(username) < 3:
        return False, "Username must be at least 3 characters long."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    if '@' not in email or '.' not in email:
        return False, "Please enter a valid email address."
        
    # Hash password
    pwd_hash = generate_password_hash(password)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, pwd_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        return True, user_id
    except sqlite3.IntegrityError as e:
        err_msg = str(e).lower()
        if "username" in err_msg:
            return False, "Username is already taken."
        elif "email" in err_msg:
            return False, "Email address is already registered."
        else:
            return False, "User with these details already exists."
    finally:
        conn.close()

def verify_user(username_or_email, password):
    """
    Verifies user credentials.
    
    Args:
        username_or_email: The input username or email
        password: The plain text password
        
    Returns:
        User record dictionary if valid, else None.
    """
    if not username_or_email or not password:
        return None
        
    username_or_email = username_or_email.strip()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    # Check matching username or email
    cursor.execute(
        "SELECT * FROM users WHERE username = ? OR email = ?", 
        (username_or_email, username_or_email.lower())
    )
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user['password_hash'], password):
        return dict(user)
    return None

def get_user_by_id(user_id):
    """
    Retrieves user by ID.
    
    Args:
        user_id: User's primary key ID
        
    Returns:
        User record dictionary if found, else None.
    """
    if user_id is None:
        return None
        
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return dict(user)
    return None
