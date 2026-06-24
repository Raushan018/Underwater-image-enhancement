from __future__ import annotations

import io
import os
import re
import base64
import json
import logging
import time
import traceback

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, send_file, Response
from flask.json.provider import DefaultJSONProvider

from werkzeug.utils import secure_filename


class NumpyJSONProvider(DefaultJSONProvider):
    """Serialize numpy scalars/arrays so jsonify never raises TypeError."""
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

# RADS-LPF pipeline
from pipeline.rads_lpf import run_pipeline

# Analysis modules
from analysis.depth import estimate_depth as rads_estimate_depth
from analysis.metrics import compute_uciqe, compute_entropy
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

# Suppress Werkzeug's development server warning
class WerkzeugFilter(logging.Filter):
    def filter(self, record):
        return "use a production wsgi server" not in record.getMessage().lower()

logging.getLogger('werkzeug').addFilter(WerkzeugFilter())

# Detection — lazy-load to stay within memory limits on constrained hosts
enable_detection = os.environ.get('ENABLE_DETECTION', 'true').lower() == 'true'
if os.environ.get('RENDER') == 'true' and 'ENABLE_DETECTION' not in os.environ:
    enable_detection = False
    logger.info("Running on Render: auto-disabling object detection to fit within the 512MB memory limit.")

_detect_objects = None
if enable_detection:
    try:
        from analysis.detection import detect_objects as _detect_objects_fn
        _detect_objects = _detect_objects_fn
        logger.info("RADS-LPF detection module loaded.")
    except Exception as e:
        logger.error(f"Failed to load detection module: {e}")
else:
    logger.info("Object detection is disabled (low-memory mode).")

app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)
app.config['SECRET_KEY'] = 'dev-super-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/images/uploads'
app.config['RESULT_FOLDER'] = 'static/images/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

UPLOAD_DIR = "/tmp/rads_lpf_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_pipeline(image_path, filename, config=None):
    """
    Runs the RADS-LPF pipeline (Code.md implementation).
    Preserves the same response structure the frontend JS expects.
    """
    start_time = time.time()
    logger.info(f"Starting RADS-LPF processing for {filename}")

    if config is None:
        config = {}

    try:
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not decode image at {image_path}")
        if original.size == 0:
            raise ValueError("Input image is empty.")

        if original.dtype != np.uint8:
            original = original.astype(np.uint8)

        # Map frontend config keys → RADS-LPF stage toggles
        stages_enabled = {
            "stage1":  config.get('stretch', True),
            "stage2":  config.get('wb', True),
            "stage3":  True,                          # DACR always on
            "stage4":  config.get('gamma', True),
            "stage5":  config.get('sharp', True),
            "stage5a": True,
            "stage5b": True,
        }

        result = run_pipeline(original, stages_enabled=stages_enabled)
        enhanced     = result["enhanced"]
        intermediates = result["intermediates"]
        rho          = result["rho"]
        depth_tier   = result["depth_tier"]

        # ── Save helpers ──────────────────────────────────────────────────────
        base_name = os.path.splitext(filename)[0]
        results = {}
        results['original'] = f"static/images/uploads/{filename}"

        def secure_save(img, suffix, ext='.jpg'):
            if img is None:
                return None
            try:
                name = f"{base_name}_{suffix}{ext}"
                path = os.path.join(app.config['RESULT_FOLDER'], name)
                cv2.imwrite(path, img)
                return name
            except Exception as save_err:
                logger.error(f"Failed to save {suffix}: {save_err}")
                return None

        # Map RADS-LPF intermediates → frontend result keys
        results['wb']    = secure_save(intermediates.get("stage2_colour"),  '1_adaptive_color')
        results['gamma'] = secure_save(intermediates.get("stage4_gamma"),   '2_adaptive_gamma')
        results['sharp'] = secure_save(intermediates.get("stage5_denoise"), '3_edge_preserving')
        results['clahe'] = secure_save(intermediates.get("stage5a_clahe"),  '4_clahe')
        results['hist']  = secure_save(intermediates.get("stage5b_he"),     '5_hist_linear')
        results['final'] = secure_save(enhanced, '6_final')

        # Base64 encode final for inline display
        try:
            is_success, buffer = cv2.imencode('.jpg', enhanced)
            if not is_success:
                raise ValueError("cv2.imencode returned False")
            results['final_b64'] = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Base64 Error: {e}")
            results['final_b64'] = None

        # ── Depth ─────────────────────────────────────────────────────────────
        metrics = {'UCIQE': 0, 'Entropy': 0}
        try:
            depth_info = rads_estimate_depth(enhanced)
            metrics['Depth']     = depth_info["tier_label"]
            metrics['DepthConf'] = round(rho, 4)
            results['depth_map'] = secure_save(depth_info["heatmap"], 'depth_color')
        except Exception as e:
            logger.error(f"Depth Error: {e}")

        # ── Detection ─────────────────────────────────────────────────────────
        results['detections'] = []
        results['composition'] = {}
        if _detect_objects is not None:
            try:
                det_result = _detect_objects(enhanced)
                results['detections']  = det_result["detections"]
                results['composition'] = det_result["composition"]
            except Exception as e:
                logger.error(f"Detection Error: {e}")

        # ── Metrics ───────────────────────────────────────────────────────────
        try:
            metrics['UCIQE']   = round(float(compute_uciqe(enhanced)), 4)
            metrics['Entropy'] = round(float(compute_entropy(enhanced)), 4)
        except Exception as e:
            logger.error(f"Metrics Error: {e}")

        # ── Histograms (Code.md §app.py — BGR 256-bin) ───────────────────────
        try:
            def _bgr_histogram(img):
                hist_data = {}
                for ch, name in enumerate(["b", "g", "r"]):
                    h, _ = np.histogram(img[:, :, ch], bins=256, range=(0, 256))
                    hist_data[name] = h.tolist()
                return hist_data

            results['histograms'] = {
                'before': _bgr_histogram(original),
                'after':  _bgr_histogram(enhanced),
            }
        except Exception:
            results['histograms'] = None

        elapsed = time.time() - start_time
        logger.info(f"RADS-LPF pipeline COMPLETE in {elapsed:.2f}s | depth={depth_tier} ρ={rho:.4f}")

        return results, metrics

    except Exception as fatal_e:
        logger.error(f"FATAL PIPELINE CRASH: {fatal_e}")
        logger.error(traceback.format_exc())
        raise fatal_e


# ── Routes ────────────────────────────────────────────────────────────────────

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

            config_str = request.form.get('config', '{}')
            try:
                config = json.loads(config_str)
            except Exception:
                config = None

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
        config   = data.get('config')

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
    end   = match.group(2)
    end   = int(end) if end else file_size - 1

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

    rv = Response(generate_bytes(), 206, mimetype='video/mp4', direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    return rv


@app.route('/download/<filename>')
def download_file(filename):
    try:
        target_format = request.args.get('format', '').lower()

        file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        if not os.path.exists(file_path):
            return "File not found", 404

        current_ext = filename.rsplit('.', 1)[1].lower()
        if not target_format or target_format == current_ext:
            return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=True)

        img = cv2.imread(file_path)
        if img is None:
            return "Error reading image source", 500

        ext_map = {'png': '.png', 'jpg': '.jpg', 'jpeg': '.jpg'}
        if target_format not in ext_map:
            return "Unsupported format", 400

        success, encoded_img = cv2.imencode(ext_map[target_format], img)
        if not success:
            return "Encoding failed", 500

        img_io = io.BytesIO(encoded_img.tobytes())
        img_io.seek(0)
        new_filename = os.path.splitext(filename)[0] + ext_map[target_format]

        return send_file(
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
