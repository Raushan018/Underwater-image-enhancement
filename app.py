import os
import cv2
import numpy as np
import logging
import time
import traceback
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, session, flash
from werkzeug.utils import secure_filename

# Import DSP modules
from src.clahe import apply_clahe
from src.histogram_linearization import apply_histogram_linearization
from src.weight_maps import generate_weight_map
from src.multiscale_fusion import apply_fusion
from src.metrics import calculate_uciqe, calculate_entropy, calculate_psnr, calculate_ssim, calculate_uiqm
from src.detection import DetectionService
from src.histogram_analysis import calculate_histogram
from src.adaptive_enhancement import AdaptiveEnhancer
from src.depth_estimation import DepthEstimator

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Check if detection is enabled
enable_detection = os.environ.get('ENABLE_DETECTION', 'true').lower() == 'true'
# Auto-disable on Render to fit within the 512MB RAM limit
if os.environ.get('RENDER') == 'true' and 'ENABLE_DETECTION' not in os.environ:
    enable_detection = False
    logger.info("Running on Render: auto-disabling object detection to fit within the 512MB memory limit.")

# Initialize Detection Service (Global)
detection_service = None
if enable_detection:
    try:
        detection_service = DetectionService()
        logger.info("YOLO-World model initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize YOLO model: {e}")
else:
    logger.info("Object detection is disabled (low-memory mode).")

# Initialize Enhancers
adaptive_enhancer = AdaptiveEnhancer()
depth_estimator = DepthEstimator()

app = Flask(__name__)
# Secure secret key (in production should be env var)
app.config['SECRET_KEY'] = 'dev-super-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/images/uploads'
app.config['RESULT_FOLDER'] = 'static/images/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB Limit
app.config['MAX_IMAGE_DIMENSION'] = 1024 # Limit max dimension for processing speed

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image_smart(image, max_dim):
    """Resizes image maintaining aspect ratio if it exceeds max_dim."""
    h, w = image.shape[:2]
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        logger.info(f"Resizing image from ({w}, {h}) to ({new_w}, {new_h})")
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

import base64

def process_pipeline(image_path, filename, config=None):
    """
    Runs the complete enhancement pipeline with strict type checks and detailed logging.
    """
    start_time = time.time()
    logger.info(f"Starting processing for {filename}")

    if config is None:
        config = {'stretch': True, 'wb': True, 'gamma': True, 'sharp': True}

    try:
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not decode image at {image_path}")
        if original.size == 0:
            raise ValueError("Input image is empty.")

        # Resize for performance constraint
        original = resize_image_smart(original, app.config['MAX_IMAGE_DIMENSION'])
        
        results = {}
        # Ensure we start with uint8
        if original.dtype != np.uint8:
            original = original.astype(np.uint8)

        current_img = original.copy()
        
        # --- Strict Validation Helper ---
        def validate_stage(img, stage_name):
            if img is None:
                 logger.error(f"STAGE FAILED: {stage_name} returned None")
                 raise ValueError(f"Pipeline failed at {stage_name} (None result)")
            
            # Check for empty
            if img.size == 0:
                 logger.error(f"STAGE FAILED: {stage_name} returned empty image")
                 raise ValueError(f"Pipeline failed at {stage_name} (Empty result)")

            # Check for valid type and convert if necessary
            if img.dtype != np.uint8:
                logger.warning(f"Stage {stage_name} returned {img.dtype}, converting to uint8")
                # Normalize if float 0-1? Or just clip? Assuming standard 0-255 range if float.
                if img.dtype == np.float32 or img.dtype == np.float64:
                     if img.max() <= 1.0:
                         img = (img * 255).astype(np.uint8)
                     else:
                         img = np.clip(img, 0, 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            logger.info(f"Stage SUCCESS: {stage_name} | Shape: {img.shape} | Mean: {np.mean(img):.2f}")
            return img

        # 1. Adaptive Histogram Stretching
        if config.get('stretch', True):
            try:
                temp_img = adaptive_enhancer.adaptive_histogram_stretching(current_img)
                current_img = validate_stage(temp_img, "Histogram Stretching")
            except Exception as e:
                logger.error(f"Error in Histogram Stretching: {e}")
                logger.error(traceback.format_exc())

        # 2. Adaptive Color Correction
        if config.get('wb', True):
            try:
                temp_img = adaptive_enhancer.adaptive_color_correction(current_img)
                current_img = validate_stage(temp_img, "Color Correction")
            except Exception as e:
                logger.error(f"Error in Color Correction: {e}")
                logger.error(traceback.format_exc())

        # 3. Adaptive Gamma Correction
        if config.get('gamma', True):
            try:
                temp_img = adaptive_enhancer.adaptive_gamma_correction(current_img)
                current_img = validate_stage(temp_img, "Gamma Correction")
            except Exception as e:
                logger.error(f"Error in Gamma Correction: {e}")
                logger.error(traceback.format_exc())
        
        # 4. Edge Preserving Filter
        edge_img = current_img # Snapshot for fusion
        if config.get('sharp', True):
            try:
                temp_img = adaptive_enhancer.apply_edge_preserving_filter(current_img)
                edge_img = validate_stage(temp_img, "Edge Preserving")
            except Exception as e:
                logger.error(f"Error in Edge Preserving: {e}")
                logger.error(traceback.format_exc())
        
        # --- Fusion Pipeline ---
        final_img = edge_img # Fallback
        
        try:
            # 5. CLAHE (Path A) - Expects uint8 (validated)
            clahe_img = apply_clahe(edge_img)
            clahe_img = validate_stage(clahe_img, "CLAHE")
            
            # 6. HE (Path B) - Expects uint8
            hist_img = apply_histogram_linearization(edge_img)
            hist_img = validate_stage(hist_img, "Histogram Linearization")
            
            # 7. Weight Maps
            w_clahe = generate_weight_map(clahe_img)
            w_hist = generate_weight_map(hist_img)
            
            # 8. Fusion
            fused_temp = apply_fusion(clahe_img, hist_img, w_clahe, w_hist)
            final_img = validate_stage(fused_temp, "Multiscale Fusion")
            
        except Exception as e:
            logger.error(f"Error in Fusion Logic: {e}")
            logger.error(traceback.format_exc())
            # Fallback to edge_img (already validated)

        # Save Logic
        base_name = os.path.splitext(filename)[0]
        results['original'] = f"static/images/uploads/{filename}"

        def secure_save(img, suffix):
            if img is None: return None
            try:
                name = f"{base_name}_{suffix}.jpg"
                path = os.path.join(app.config['RESULT_FOLDER'], name)
                cv2.imwrite(path, img)
                return name
            except Exception as save_err:
                logger.error(f"Failed to save {suffix}: {save_err}")
                return None

        results['wb'] = secure_save(current_img, '1_adaptive_color')
        results['gamma'] = secure_save(current_img, '2_adaptive_gamma')
        results['sharp'] = secure_save(edge_img, '3_edge_preserving')
        
        if 'clahe_img' in locals(): results['clahe'] = secure_save(clahe_img, '4_clahe')
        if 'hist_img' in locals(): results['hist'] = secure_save(hist_img, '5_hist_linear')
        
        results['final'] = secure_save(final_img, '6_final')

        # Base64 Encode
        try:
            is_success, buffer = cv2.imencode('.jpg', final_img)
            if not is_success:
                raise ValueError("cv2.imencode returned False")
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            results['final_b64'] = f"data:image/jpeg;base64,{jpg_as_text}"
        except Exception as e:
            logger.error(f"Base64 Error: {e}")
            results['final_b64'] = None # Frontend should handle this fall back to URL

        # Metrics & Extra Features (Safe wrappers)
        metrics = {'UCIQE': 0, 'Entropy': 0, 'UIQM (Est)': 0}
        try:
            metrics['UCIQE'] = round(float(calculate_uciqe(final_img)), 4)
            metrics['Entropy'] = round(float(calculate_entropy(final_img)), 4)
            metrics['UIQM (Est)'] = round(float(calculate_uiqm(final_img)), 4)
        except Exception as e:
            logger.error(f"Metrics Error: {e}")

        # Depth
        try:
            depth_info = depth_estimator.estimate_depth(original)
            metrics['Depth'] = depth_info.get('depth_range', 'N/A')
            metrics['DepthConf'] = depth_info.get('confidence', 0)
            
            d_maps = depth_estimator.generate_depth_map(original)
            if d_maps:
                results['depth_map'] = secure_save(d_maps['heatmap'], 'depth_color')
                # Raw depth save
                raw_name = f"{base_name}_depth_raw.png"
                cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], raw_name), d_maps['raw_depth'])
                results['depth_raw'] = raw_name
        except Exception as e:
             logger.error(f"Depth Error: {e}")

        # Detection
        results['detections'] = []
        results['composition'] = {}
        if detection_service:
            try:
                dets, comp = detection_service.detect_objects(final_img)
                results['detections'] = dets
                results['composition'] = comp
            except Exception as e:
                logger.error(f"Detection Error: {e}")

        # Histograms
        try:
             results['histograms'] = {
                'before': calculate_histogram(original),
                'after': calculate_histogram(final_img)
             }
        except:
             results['histograms'] = None
             
        elapsed = time.time() - start_time
        logger.info(f"Pipeline COMPLETE in {elapsed:.2f}s")
        
        return results, metrics

    except Exception as fatal_e:
        logger.error(f"FATAL PIPELINE CRASH: {fatal_e}")
        logger.error(traceback.format_exc())
        raise fatal_e

@app.route('/')
def index():
    return render_template('index.html', detection_enabled=enable_detection)

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'success': False, 'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)}MB.'}), 413

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part in request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            logger.info(f"Receiving file: {filename}")
            file.save(filepath)
            
            # Parse config from Form Data
            config_str = request.form.get('config', '{}')
            try:
                config = json.loads(config_str)
            except:
                config = None
                
            # Process
            try:
                image_results, metrics = process_pipeline(filepath, filename, config)
                return jsonify({
                    'success': True,
                    'images': image_results,
                    'metrics': metrics,
                    'filename': filename 
                })
            except Exception as e:
                logger.error(f"Pipeline Exception during upload: {e}")
                return jsonify({'success': False, 'error': f"Processing failed: {str(e)}"}), 500
                
        return jsonify({'success': False, 'error': 'Invalid file type. Allowed: JPG, PNG, BMP, WEBP'}), 400
        
    except Exception as server_error:
        logger.error(f"Server upload error: {server_error}")
        return jsonify({'success': False, 'error': 'Internal server error during upload'}), 500

@app.route('/reprocess', methods=['POST'])
def reprocess_image():
    try:
        data = request.json
        filename = data.get('filename')
        config = data.get('config')
        
        if not filename:
             return jsonify({'success': False, 'error': 'No filename provided'}), 400
             
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'}), 404
            
        try:
            image_results, metrics = process_pipeline(filepath, filename, config)
            return jsonify({
                'success': True,
                'images': image_results,
                'metrics': metrics,
                'filename': filename
            })
        except Exception as e:
            return jsonify({'success': False, 'error': f"Reprocessing failed: {str(e)}"}), 500
            
    except Exception as e:
         return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/video/<path:filename>')
def serve_video(filename):
    import re
    from flask import Response
    video_dir = os.path.join(app.root_path, 'static', 'video')
    file_path = os.path.join(video_dir, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
        
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_from_directory(video_dir, filename)
        
    match = re.search(r'bytes=(\d+)-(\d*)', range_header)
    if not match:
        return send_from_directory(video_dir, filename)
        
    start = int(match.group(1))
    end = match.group(2)
    end = int(end) if end else file_size - 1
    
    if start >= file_size or end >= file_size:
        return "Requested range not satisfiable", 416
        
    length = end - start + 1
    
    def generate_bytes():
        with open(file_path, 'rb') as f:
            f.seek(start)
            remaining = length
            chunk_size = 1024 * 1024
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                data = f.read(to_read)
                if not data:
                    break
                remaining -= len(data)
                yield data

    rv = Response(generate_bytes(),
                  206,
                  mimetype='video/mp4',
                  direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    return rv

@app.route('/download/<filename>')
def download_file(filename):
    """
    Downloads file with optional format conversion.
    Query param: ?format=png|jpg
    """
    try:
        target_format = request.args.get('format', '').lower()
        
        # Secure path setup
        file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        if not os.path.exists(file_path):
            return "File not found", 404

        # If no specific format requested or same format, just send file
        current_ext = filename.rsplit('.', 1)[1].lower()
        if not target_format or target_format == current_ext:
            return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=True)

        # Conversion logic
        img = cv2.imread(file_path)
        if img is None:
            return "Error reading image source", 500
        
        # Encode to target format
        ext_map = {'png': '.png', 'jpg': '.jpg', 'jpeg': '.jpg'}
        if target_format not in ext_map:
            return "Unsupported format", 400
            
        success, encoded_img = cv2.imencode(ext_map[target_format], img)
        if not success:
            return "Encoding failed", 500
            
        # Create byte stream
        from io import BytesIO
        from flask import send_file as flask_send_file
        
        img_io = BytesIO(encoded_img.tobytes())
        img_io.seek(0)
        
        new_filename = os.path.splitext(filename)[0] + ext_map[target_format]
        
        return flask_send_file(
            img_io,
            mimetype=f'image/{target_format if target_format != "jpg" else "jpeg"}',
            as_attachment=True,
            download_name=new_filename
        )
            
    except Exception as e:
        logger.error(f"Download error: {e}")
        return str(e), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    is_prod = os.environ.get('FLASK_ENV') == 'production'
    app.run(host='0.0.0.0', port=port, debug=not is_prod, use_reloader=False)
