"""
Web Routes for HTML pages
Serves the frontend web interface
"""
from flask import Blueprint, render_template, redirect, url_for
from routes.auth import login_required

web_bp = Blueprint('web', __name__)


@web_bp.route('/')
@login_required
def index():
    """Main dashboard"""
    return render_template('index.html')


@web_bp.route('/detect')
@login_required
def detect():
    """Disease detection interface"""
    return render_template('detect.html')


@web_bp.route('/visualize')
@login_required
def visualize():
    """3D brain visualization page"""
    from flask import request
    patient_id = request.args.get('patient_id') or request.args.get('type') or 'patient_001'
    return render_template('visualize.html', patient_id=patient_id)


@web_bp.route('/patient/<patient_id>')
@login_required
def patient_detail(patient_id):
    """Patient detail view"""
    return render_template('patient.html', patient_id=patient_id)


@web_bp.route('/simulation')
@login_required
def simulation():
    """Neural simulation interface"""
    return render_template('simulation.html')


@web_bp.route('/about')
def about():
    """About page"""
    return render_template('about.html')

