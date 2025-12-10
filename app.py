import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, session, flash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from captcha.image import ImageCaptcha
import random
import string
from src.models import db, User

# Import DSP modules
from src.white_balance import apply_white_balance
from src.gamma_correction import apply_gamma_correction
from src.sharpening import apply_sharpening
from src.clahe import apply_clahe
from src.histogram_linearization import apply_histogram_linearization
from src.weight_maps import generate_weight_map
from src.multiscale_fusion import apply_fusion
from src.multiscale_fusion import apply_fusion
from src.metrics import calculate_uciqe, calculate_entropy, calculate_psnr, calculate_ssim, calculate_uiqm
from src.detection import DetectionService
from src.histogram_analysis import calculate_histogram

# Initialize Detection Service (Global to avoid reloading)
# We use try/except block to avoid crashing if weights fail to download immediately on start? 
# Usually better to load lazily or just load.
detection_service = None
try:
    detection_service = DetectionService()
    print("YOLO-World model initialized successfully.")
except Exception as e:
    print(f"Failed to initialize YOLO model: {e}")

app = Flask(__name__)
# Secure secret key (in production should be env var)
app.config['SECRET_KEY'] = 'dev-super-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'static/images/uploads'
app.config['RESULT_FOLDER'] = 'static/images/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Init DB
db.init_app(app)

# Init Login
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create tables
with app.app_context():
    db.create_all()

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# --- Auth Routes ---
@app.route('/captcha-image')
def captcha_image():
    image = ImageCaptcha(width=280, height=90)
    captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    session['captcha'] = captcha_text
    data = image.generate(captcha_text)
    from io import BytesIO
    from flask import send_file
    return send_file(BytesIO(data.read()), mimetype='image/png')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        captcha_input = request.form.get('captcha', '').upper()
        
        # Verify Captcha
        if 'captcha' not in session or session['captcha'] != captcha_input:
            flash('Invalid Captcha', 'error')
            return render_template('login.html')
            
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'error')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')
        captcha_input = request.form.get('captcha', '').upper()
        
        if 'captcha' not in session or session['captcha'] != captcha_input:
            flash('Invalid Captcha', 'error')
            return render_template('register.html')
            
        if password != confirm:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
            
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return render_template('register.html')
            
        new_user = User(email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        return redirect(url_for('index'))
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pipeline(image_path, filename):
    """
    Runs the complete enhancement pipeline and returns a dictionary of results.
    """
    # Load original image
    original = cv2.imread(image_path)
    base_name = os.path.splitext(filename)[0]
    
    results = {}
    
    # 1. White Balancing
    wb_img = apply_white_balance(original)
    
    # 2. Gamma Correction
    gamma_img = apply_gamma_correction(wb_img)
    
    # 3. Sharpening
    sharp_img = apply_sharpening(gamma_img)
    
    # 4. Path A: Luminance Enhancement (CLAHE)
    clahe_img = apply_clahe(sharp_img)
    
    # 5. Path B: Histogram Linearization
    hist_img = apply_histogram_linearization(sharp_img)
    
    # 6. Weight Maps
    w_clahe = generate_weight_map(clahe_img)
    w_hist = generate_weight_map(hist_img)
    
    # 7. Multiscale Fusion
    final_img = apply_fusion(clahe_img, hist_img, w_clahe, w_hist)
    
    # Save Images
    def save_img(img, suffix):
        name = f"{base_name}_{suffix}.jpg"
        path = os.path.join(app.config['RESULT_FOLDER'], name)
        cv2.imwrite(path, img)
        return name

    results['original'] = f"static/images/uploads/{filename}"
    # Flask serves static from /static/...
    # Let's verify where uploads are. They are in static/images/uploads.
    # So URL is /static/images/uploads/filename
    
    # We will return filenames and let the frontend construct paths or return full static paths.
    results['wb'] = save_img(wb_img, '1_white_balance')
    results['gamma'] = save_img(gamma_img, '2_gamma')
    results['sharp'] = save_img(sharp_img, '3_sharpened')
    results['clahe'] = save_img(clahe_img, '4_clahe')
    results['hist'] = save_img(hist_img, '5_hist_linear')
    results['final'] = save_img(final_img, '6_final')
    
    # 8. Compute Metrics
    metrics = {
        'UCIQE': round(float(calculate_uciqe(final_img)), 4),
        'Entropy': round(float(calculate_entropy(final_img)), 4),
        'UIQM (Est)': round(float(calculate_uiqm(final_img)), 4)
    }
    
    # 9. Object Detection
    detections = []
    if detection_service:
        try:
            # Detect on the final enhanced image
            detections = detection_service.detect_objects(final_img)
        except Exception as e:
            print(f"Detection error: {e}")
    
            print(f"Detection error: {e}")
    
    results['detections'] = detections
    
    # 10. Histograms
    try:
        results['histograms'] = {
            'before': calculate_histogram(original),
            'after': calculate_histogram(final_img)
        }
    except Exception as e:
        print(f"Histogram error: {e}")
        results['histograms'] = None

    # PSNR/SSIM require ground truth. If original is treated as reference (which it isn't for enhancement),
    # these values are meaningless basically. But often requested in papers to compare against raw.
    # Or if we had a GT. We don't have GT. 
    # Usually we compare enhanced vs original to see how much it changed (MSE), but PSNR is "quality" vs "noise".
    # I'll output them comparing Final vs Original just for "difference" measurement, 
    # but label them clearly or omit if confusing. 
    # The prompt says: "PSNR/SSIM (if ground truth available)".
    # We will leave them 0 or None if no GT.
    metrics['PSNR'] = "N/A"
    metrics['SSIM'] = "N/A"
    
    return results, metrics

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            image_results, metrics = process_pipeline(filepath, filename)
            return jsonify({
                'success': True,
                'images': image_results,
                'metrics': metrics
            })
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<filename>')
@login_required
def download_file(filename):
    """
    Downloads file with optional format conversion.
    Query param: ?format=png|jpg
    """
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
    try:
        # Read image
        img = cv2.imread(file_path)
        if img is None:
            return "Error reading image", 500
        
        # Encode to target format
        # mapping format to extension
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
        print(f"Conversion error: {e}")
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
