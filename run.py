"""
Virtual Brain Inference - Application Launcher
Simple script to run the Flask application
"""
import os
from app import create_app

if __name__ == '__main__':
    # Set environment
    env = os.environ.get('FLASK_ENV', 'development')
    
    # Create and run app
    app = create_app(env)
    
    print("=" * 60)
    print("Virtual Brain Inference - Starting Application")
    print("=" * 60)
    print(f"Environment: {env}")
    print(f"Debug Mode: {app.config['DEBUG']}")
    print(f"Server: http://localhost:5000")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    )
