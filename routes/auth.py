"""
Authentication Routes and Helpers for Virtual Brain Inference.
Handles user registration, login, logout, and route protection.
"""
from flask import Blueprint, request, render_template, redirect, url_for, flash, session
from functools import wraps
from utils.auth_db import create_user, verify_user, get_user_by_id

auth_bp = Blueprint('auth', __name__)

def login_required(f):
    """
    Decorator to protect routes requiring authentication.
    Redirects unauthenticated users to the login page.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "error")
            return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.app_context_processor
def inject_user():
    """
    Context processor to inject the current logged-in user into all templates.
    Allows accessing {{ current_user }} in HTML files.
    """
    user_id = session.get('user_id')
    if user_id:
        user = get_user_by_id(user_id)
        if user:
            return dict(current_user=user)
    return dict(current_user=None)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if 'user_id' in session:
        return redirect(url_for('web.index'))
        
    if request.method == 'POST':
        username_or_email = request.form.get('username')
        password = request.form.get('password')
        
        user = verify_user(username_or_email, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f"Welcome back, {user['username']}!", "success")
            
            # Redirect to next page if specified
            next_url = request.args.get('next')
            if next_url and next_url.startswith('/'):
                return redirect(next_url)
            return redirect(url_for('web.index'))
        else:
            flash("Invalid username/email or password.", "error")
            
    return render_template('login.html')

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handles user registration."""
    if 'user_id' in session:
        return redirect(url_for('web.index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template('signup.html')
            
        success, result = create_user(username, email, password)
        if success:
            # Auto login after successful registration
            session['user_id'] = result
            session['username'] = username.strip()
            flash("Registration successful! Welcome to Virtual Brain Inference.", "success")
            return redirect(url_for('web.index'))
        else:
            flash(result, "error")
            
    return render_template('signup.html')

@auth_bp.route('/logout')
def logout():
    """Logs out the user and clears session data."""
    session.pop('user_id', None)
    session.pop('username', None)
    flash("You have been logged out successfully.", "info")
    return redirect(url_for('auth.login'))
