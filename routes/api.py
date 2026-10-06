"""
API Routes for Virtual Brain Inference
REST endpoints for brain analysis and visualization
"""
from flask import Blueprint, request, jsonify, send_file
from pathlib import Path
import numpy as np
import json
from werkzeug.utils import secure_filename
import traceback

from config import Config
from models.disease_detector import DiseaseDetector
from models.data_processor import DataProcessor
from simulation.connectome import Connectome
from simulation.neural_mass import NeuralMassModel
from simulation.bayesian_inference import BayesianInference
from visualization.brain_mesh import BrainMesh
from visualization.vessel_network import VesselNetwork
from visualization.blockage_detector import BlockageDetector

api_bp = Blueprint('api', __name__)
config = Config()

# Initialize components
disease_detector = DiseaseDetector()
data_processor = DataProcessor()


@api_bp.route('/detect', methods=['POST'])
def detect_disease():
    """
    Disease detection endpoint
    Expects: uploaded brain imaging file or JSON data
    Returns: disease prediction with probabilities
    """
    try:
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            upload_path = config.UPLOAD_FOLDER / filename
            file.save(upload_path)
            
            # Process the imaging data
            brain_data = data_processor.preprocess(upload_path)
            
        elif request.is_json:
            # Receive synthetic/test data as JSON
            data = request.get_json()
            
            if 'brain_data' in data:
                brain_data = np.array(data['brain_data'])
            else:
                # Generate synthetic data for testing
                brain_data = data_processor.create_synthetic_data()
        else:
            # No data provided - use synthetic
            brain_data = data_processor.create_synthetic_data()
        
        # Run disease detection
        prediction = disease_detector.predict(brain_data)
        
        # Get feature importance
        features = disease_detector.get_feature_importance(brain_data)
        
        response = {
            'success': True,
            'prediction': prediction,
            'feature_importance': features,
            'data_shape': brain_data.shape if hasattr(brain_data, 'shape') else None
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@api_bp.route('/visualization/<patient_id>', methods=['GET'])
def get_visualization_data(patient_id):
    """
    Get 3D visualization data for a patient
    Returns: brain mesh, vessel network, and blockage data
    """
    try:
        # Get query parameters for localized damage
        f_x = request.args.get('focal_x', type=float)
        f_y = request.args.get('focal_y', type=float)
        f_z = request.args.get('focal_z', type=float)
        radius = request.args.get('radius', 0.4, type=float)
        
        custom_focal = np.array([f_x, f_y, f_z]) if f_x is not None else None
        
        # Generate or load brain mesh
        brain_mesh = BrainMesh()
        brain_mesh.generate_synthetic_brain(resolution=40)
        brain_mesh.parcellate(num_regions=90)
        
        # Generate vessel network
        vessel_network = VesselNetwork()
        vessel_network.generate_synthetic_network(num_vessels=100, network_type='mixed')
        vessel_network.calculate_flow()
        
        # Detect blockages
        blockage_detector = BlockageDetector(sensitivity=0.7)
        blockage_analysis = blockage_detector.analyze_network(
            vessel_network.vessels,
            vessel_network.flow_rates
        )
        
        # Apply disease effects if applicable
        disease_key = patient_id.lower()
        is_tumor = any(t in disease_key for t in ['tumor', 'glioma', 'meningioma', 'pituitary'])
        
        if is_tumor or custom_focal is not None:
            # Determine which deformation to apply
            deform_type = 'tumor'
            if 'glioma' in disease_key: deform_type = 'tumor' # could be specific later
            
            # Apply deformation and get focal point
            deform_meta = brain_mesh.apply_disease_deformation(deform_type, severity=0.8, focal_point=custom_focal)
            f_point = deform_meta.get('center')
            f_radius = radius if custom_focal is not None else deform_meta.get('radius', 0.4)
            
            if f_point:
                # 1. Apply spatial damage to vessels (nerve/vessel damage)
                vessel_network.apply_spatial_damage(f_point, f_radius, max_severity=0.9)
                
                # 2. Re-detect blockages to reflect the spatial damage
                blockage_analysis = blockage_detector.analyze_network(
                    vessel_network.vessels,
                    vessel_network.flow_rates
                )
                
                # 3. Highlight impacted brain regions
                affected_regions = []
                temp_conn = Connectome(num_regions=90)
                temp_conn.generate_synthetic()
                for i, pos in enumerate(temp_conn.region_positions):
                    if np.linalg.norm(pos - np.array(f_point)) < f_radius:
                        affected_regions.append(i)
                
                if affected_regions:
                    brain_mesh.add_region('Tumor Impact Area', affected_regions, (1.0, 0.2, 0.2))
        
        response = {
            'success': True,
            'patient_id': patient_id,
            'brain_mesh': brain_mesh.export_to_threejs(),
            'vessel_network': vessel_network.export_to_threejs(),
            'blockage_analysis': blockage_analysis,
            'network_stats': vessel_network.get_network_statistics()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@api_bp.route('/detect_tumor_spatial', methods=['POST'])
def detect_tumor_spatial():
    """
    Detect tumor and estimate its 3D coordinates from an image
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        filename = secure_filename(file.filename)
        upload_path = config.UPLOAD_FOLDER / filename
        file.save(upload_path)
        
        # Process and predict
        brain_data = data_processor.preprocess(upload_path)
        prediction = disease_detector.predict(brain_data)
        
        tumor_classes = ['Glioma', 'Meningioma', 'Pituitary', 'Tumor']
        if prediction['predicted_disease'] in tumor_classes:
            localization = disease_detector.get_tumor_localization(brain_data)
            return jsonify({
                'success': True,
                'is_tumor': True,
                'localization': localization,
                'prediction': prediction
            }), 200
        else:
            return jsonify({
                'success': True,
                'is_tumor': False,
                'prediction': prediction
            }), 200
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/simulate', methods=['POST'])
def simulate_neural_activity():
    """
    Simulate neural activity using Virtual Brain model
    Expects: simulation parameters
    Returns: simulated time series and functional connectivity
    """
    try:
        data = request.get_json() if request.is_json else {}
        
        # Get parameters
        num_regions = data.get('num_regions', 90)
        duration = data.get('duration', 5.0)  # seconds
        disease_type = data.get('disease_type', 'healthy')
        
        # Create connectome
        connectome = Connectome(num_regions=num_regions)
        connectome.generate_synthetic(connectivity_density=0.3)
        
        # Create neural mass model
        neural_model = NeuralMassModel(num_regions=num_regions, dt=0.001)
        
        # Simulate
        time, activity = neural_model.simulate_disease_condition(
            duration=duration,
            connectivity_matrix=connectome.get_weighted_matrix(),
            disease_type=disease_type
        )
        
        # Calculate functional connectivity
        fc = neural_model.calculate_functional_connectivity(activity)
        
        # Analyze one region for oscillations
        region_activity = activity[:, 0]
        oscillation_analysis = neural_model.analyze_oscillations(time, region_activity)
        
        # Downsample for transmission (every 10th point)
        time_downsampled = time[::10].tolist()
        activity_downsampled = activity[::10, ::2].tolist()  # Every 10th time, every 2nd region
        
        response = {
            'success': True,
            'time': time_downsampled,
            'activity': activity_downsampled,
            'functional_connectivity': fc.tolist(),
            'structural_connectivity': connectome.get_weighted_matrix().tolist(),
            'oscillation_analysis': oscillation_analysis,
            'network_metrics': connectome.get_network_metrics(),
            'simulation_params': {
                'num_regions': num_regions,
                'duration': duration,
                'disease_type': disease_type
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@api_bp.route('/connectome/<patient_id>', methods=['GET'])
def get_connectome(patient_id):
    """
    Get structural connectome data
    Returns: connectivity matrix and network metrics
    """
    try:
        # Generate or load connectome
        connectome = Connectome(num_regions=90)
        connectome.generate_synthetic(connectivity_density=0.3)
        
        # Get network properties
        metrics = connectome.get_network_metrics()
        hubs = connectome.get_hub_regions(top_k=10)
        modules = connectome.get_modular_structure()
        
        response = {
            'success': True,
            'patient_id': patient_id,
            'connectivity_matrix': connectome.get_weighted_matrix().tolist(),
            'region_names': connectome.region_names,
            'region_positions': connectome.region_positions.tolist(),
            'network_metrics': metrics,
            'hub_regions': hubs,
            'modules': modules
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/upload', methods=['POST'])
def upload_imaging_data():
    """
    Upload brain imaging data
    Returns: upload confirmation and data preview
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        filename = secure_filename(file.filename)
        if filename.lower().endswith('.nii.gz'):
            file_ext = 'nii.gz'
        else:
            file_ext = Path(filename).suffix.lower().lstrip('.')
        
        if file_ext not in config.ALLOWED_EXTENSIONS:
            return jsonify({
                'error': f'File type not allowed. Allowed types: {config.ALLOWED_EXTENSIONS}'
            }), 400
        
        # Save file
        upload_path = config.UPLOAD_FOLDER / filename
        file.save(upload_path)
        
        # Extract metadata
        try:
            if file_ext in ['nii', 'gz', 'nii.gz']:
                brain_data = data_processor.load_nifti(upload_path)
            elif file_ext == 'dcm':
                brain_data = data_processor.load_dicom(upload_path)
            else:
                brain_data = data_processor.load_image(upload_path)
            
            features = data_processor.extract_features(brain_data)
        except Exception as e:
            features = {}
        
        response = {
            'success': True,
            'filename': filename,
            'file_size': upload_path.stat().st_size,
            'upload_path': str(upload_path),
            'features': features
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/blockages/<patient_id>', methods=['GET'])
def get_blockage_analysis(patient_id):
    """
    Get detailed blockage analysis
    Returns: blockage locations, severity, and recommendations
    """
    try:
        # Generate vessel network with blockages
        vessel_network = VesselNetwork()
        vessel_network.generate_synthetic_network(num_vessels=100)
        vessel_network.calculate_flow()
        
        # Add synthetic blockages
        vessel_network.add_blockage(15, 0.7)
        vessel_network.add_blockage(30, 0.85)
        vessel_network.add_blockage(50, 0.5)
        
        # Analyze blockages
        detector = BlockageDetector(sensitivity=0.7)
        analysis = detector.analyze_network(
            vessel_network.vessels,
            vessel_network.flow_rates
        )
        
        report = detector.export_report()
        
        response = {
            'success': True,
            'patient_id': patient_id,
            'analysis': analysis,
            'report': report,
            'vessels_with_blockages': [
                v for v in vessel_network.vessels 
                if v.get('blockage_severity', 0) > 0
            ]
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/inference/parameters', methods=['POST'])
def infer_parameters():
    """
    Perform Bayesian parameter inference
    Expects: observed functional connectivity data
    Returns: estimated optimal parameters
    """
    try:
        data = request.get_json()
        
        # Get observed FC (or generate synthetic)
        if 'functional_connectivity' in data:
            observed_fc = np.array(data['functional_connectivity'])
        else:
            # Generate synthetic FC for testing
            n = 90
            observed_fc = np.random.rand(n, n)
            observed_fc = (observed_fc + observed_fc.T) / 2
            np.fill_diagonal(observed_fc, 1.0)
        
        # Get structural connectivity
        if 'structural_connectivity' in data:
            structural_conn = np.array(data['structural_connectivity'])
        else:
            connectome = Connectome(num_regions=observed_fc.shape[0])
            connectome.generate_synthetic()
            structural_conn = connectome.get_weighted_matrix()
        
        # Perform inference
        inference = BayesianInference(num_params=5)
        result = inference.estimate_functional_connectivity_params(
            observed_fc,
            structural_conn
        )
        
        response = {
            'success': True,
            'estimated_parameters': result,
            'parameter_names': [
                'coupling_strength',
                'noise_level',
                'excitability',
                'inhibition',
                'time_constant'
            ]
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
