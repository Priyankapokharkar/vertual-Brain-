"""
Virtual Brain Inference - Flask Application
Main application entry point
"""
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from pathlib import Path
import os

from config import get_config

def create_app(config_name='development'):
    """Application factory pattern"""
    
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(get_config(config_name))
    
    # Enable CORS
    CORS(app)
    
    # Create necessary directories
    directories = [
        app.config['UPLOAD_FOLDER'],
        app.config['MODELS_DIR'],
        app.config['DATA_DIR'],
        Path(app.config['LOG_FILE']).parent
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Initialize users database
    from utils.auth_db import init_db
    init_db()
    
    # Register blueprints
    from routes.api import api_bp
    from routes.web import web_bp
    from routes.auth import auth_bp
    
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(web_bp)
    app.register_blueprint(auth_bp)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    # Health check endpoint
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'healthy',
            'version': app.config['API_VERSION']
        })
    
    return app


if __name__ == '__main__':
    # Get environment from environment variable
    env = os.environ.get('FLASK_ENV', 'development')
    app = create_app(env)
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=(env == 'development')
    )
