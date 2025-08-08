from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_cors import CORS
import os
import cv2
import numpy as np
import json
import base64
from datetime import datetime
from werkzeug.utils import secure_filename
import logging

from ..core.classification_engine import ClassificationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
           template_folder='../../templates',
           static_folder='../../static')
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Initialize classification engine
classification_engine = ClassificationEngine()

def convert_numpy_to_list(obj):
    """Recursively convert numpy arrays to lists"""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_list(item) for item in obj)
    else:
        return obj

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image_path):
    """Encode image to base64 for web display"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None

@app.route('/')
def index():
    """Main dashboard"""
    try:
        stats = classification_engine.get_statistics()
        classes = classification_engine.db_manager.get_classes()
        
        return render_template('index.html', stats=stats, classes=classes)
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        return render_template('error.html', error=str(e))

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    """Upload and process image"""
    if request.method == 'GET':
        return render_template('upload.html')
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process image
            image = cv2.imread(file_path)
            results = classification_engine.process_image(image, file_path)
            
            # Create annotated image
            annotated_image = classification_engine._draw_results(image, results)
            annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], f"annotated_{filename}")
            cv2.imwrite(annotated_path, annotated_image)
            
            # Prepare response
            response_data = {
                'success': True,
                'original_image': f"/static/uploads/{filename}",
                'annotated_image': f"/static/uploads/annotated_{filename}",
                'results': convert_numpy_to_list(results)
            }
            
            return jsonify(response_data)
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/classes')
def view_classes():
    """View all classes"""
    try:
        classes = classification_engine.db_manager.get_classes()
        return render_template('classes.html', classes=classes)
    except Exception as e:
        logger.error(f"Error loading classes: {e}")
        return render_template('error.html', error=str(e))

@app.route('/clustering')
def clustering_page():
    """Clustering and new class discovery page"""
    try:
        # Get unknown objects count
        stats = classification_engine.db_manager.get_database_stats()
        unknown_count = stats['num_unknown_objects']
        
        return render_template('clustering.html', unknown_count=unknown_count)
    except Exception as e:
        logger.error(f"Error loading clustering page: {e}")
        return render_template('error.html', error=str(e))

@app.route('/api/discover_classes', methods=['POST'])
def discover_classes():
    """API endpoint to discover new classes"""
    try:
        method = request.json.get('method', 'kmeans')
        results = classification_engine.discover_new_classes(method)
        
        # Convert numpy arrays to lists for JSON serialization
        
        results = convert_numpy_to_list(results)
        
        # Get sample images for clusters
        if results['status'] == 'success':
            unknown_objects = classification_engine.db_manager.get_unknown_objects()
            
            # Add sample images to results
            for cluster_id, representative_indices in results['representatives'].items():
                sample_images = []
                for idx in representative_indices[:3]:  # Max 3 samples
                    if idx < len(unknown_objects):
                        obj = unknown_objects[idx]
                        if obj['image_path'] and os.path.exists(obj['image_path']):
                            # Convert to web-accessible URL
                            web_path = f"/{obj['image_path']}" if not obj['image_path'].startswith('/') else obj['image_path']
                            sample_images.append(web_path)
                
                results['representatives'][cluster_id] = {
                    'indices': representative_indices,
                    'sample_images': sample_images
                }
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error discovering classes: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/create_class', methods=['POST'])
def create_class():
    """API endpoint to create class from cluster"""
    try:
        data = request.json
        cluster_id = data.get('cluster_id')
        class_name = data.get('class_name')
        description = data.get('description', '')
        
        if not cluster_id or not class_name:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        class_id = classification_engine.create_class_from_cluster(
            cluster_id, class_name, description
        )
        
        return jsonify({
            'success': True,
            'class_id': class_id,
            'message': f"Created class '{class_name}' successfully"
        })
        
    except Exception as e:
        logger.error(f"Error creating class: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/add_manual_class', methods=['POST'])
def add_manual_class():
    """API endpoint to manually add a class"""
    try:
        class_name = request.form.get('class_name')
        description = request.form.get('description', '')
        
        if not class_name:
            return jsonify({'error': 'Class name is required'}), 400
        
        # Handle uploaded files
        image_paths = []
        if 'files' in request.files:
            files = request.files.getlist('files')
            
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"manual_{timestamp}_{filename}"
                    
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    image_paths.append(file_path)
        
        if not image_paths:
            return jsonify({'error': 'At least one sample image is required'}), 400
        
        class_id = classification_engine.add_manual_class(
            class_name, image_paths, description
        )
        
        return jsonify({
            'success': True,
            'class_id': class_id,
            'message': f"Created class '{class_name}' with {len(image_paths)} samples"
        })
        
    except Exception as e:
        logger.error(f"Error adding manual class: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """API endpoint to get system statistics"""
    try:
        stats = classification_engine.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera_feed')
def camera_feed():
    """API endpoint for camera feed (placeholder)"""
    # This would typically stream video frames
    # For now, return a simple response
    return jsonify({
        'message': 'Camera feed endpoint - implement WebRTC or similar for real-time streaming'
    })

@app.route('/camera')
def camera_page():
    """Camera interface page"""
    return render_template('camera.html')

@app.route('/api/process_camera_frame', methods=['POST'])
def process_camera_frame():
    """Process a single camera frame"""
    try:
        # Get image data from request
        image_data = request.json.get('image_data')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image
        results = classification_engine.process_image(image, "camera_frame")
        
        # Create annotated image
        annotated_image = classification_engine._draw_results(image, results)
        
        # Encode annotated image back to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_base64 = base64.b64encode(buffer).decode()
        
        return jsonify({
            'success': True,
            'annotated_image': f"data:image/jpeg;base64,{annotated_base64}",
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error processing camera frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500

def run_app(host='0.0.0.0', port=12000, debug=True):
    """Run the Flask application"""
    logger.info(f"Starting web application on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_app()