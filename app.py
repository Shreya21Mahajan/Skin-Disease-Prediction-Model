from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import io
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from skin_model import DERSCAN_MODEL, CLASS_NAMES, IMAGE_SIZE # Assuming skin_model is fixed

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_strong_and_secret_key_here' # For session security
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_file):
    """Loads, resizes, and normalizes the image for the model."""
    img = Image.open(image_file).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img).astype('float32') / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

# Define DUMMY_USER_DATA globally for context
DUMMY_USER_DATA = {
    'name': 'Vihaan Nambiar', 
    'last_login': '2025-10-12 17:30 IST', 
    'status': 'Active',
    'notes': 'BAS Crew Wellness Check - Multimodal AI System User'
}

# --- AUTHENTICATION ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = request.args.get('error') # Retrieve error if redirected from another page
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # --- DUMMY AUTHENTICATION LOGIC ---
        if username == 'Vihaan Nambiar' and password == 'bascrew':
            # Authentication success: set session data
            session['user_data'] = DUMMY_USER_DATA
            return redirect(url_for('home'))
        else:
            # Authentication failure: Render login with specific error
            return render_template('login.html', error='Invalid Crew ID or Password.')
            
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    """Clears the session data and redirects to the login page."""
    session.pop('user_data', None)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET'])
def register():
    """Renders the new user registration/access request page."""
    # Retrieve status message from query parameters (used when redirected from submit_request failure)
    status_message = request.args.get('status') 
    return render_template('register.html', status_message=status_message)

@app.route('/submit_request', methods=['POST'])
def submit_request():
    """Handles POST request for submitting a new user access request."""
    # Retrieve data from the registration form
    full_name = request.form.get('full_name')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')

    # 1. Basic Validation: Check if passwords match
    if password != confirm_password:
        # If passwords do not match, redirect back to the registration page 
        return redirect(url_for('register', status="Error: Passwords do not match."))

    # 2. Simulate Data Logging/Processing (Success path)
    print(f"--- NEW USER ACCESS REQUEST LOGGED AND PROVISIONED ---")
    print(f"Name: {full_name}")
    print(f"Email: {email}")
    print(f"Account Provisioned and Logged In.")
    print(f"--- END REQUEST LOG ---")

    # 3. ⭐️ FIX: Simulate immediate login and set session data ⭐️
    # We now set the session data for the newly registered user
    session['user_data'] = {
        'name': full_name, # Use the name provided in the form
        'last_login': '2025-10-12 18:08 IST', # Use current time
        'status': 'Active',
        'notes': 'Wellness Check - Account Activated'
    }
    
    # 4. Redirect to Home Page
    # The home route will now find the 'user_data' in the session and display the profile.
    return redirect(url_for('home'))

# --- APPLICATION ROUTES ---

@app.route('/')
def home():
    """Checks for user login status and serves the home page."""
    user_data = session.get('user_data')
    
    if user_data is None:
        return redirect(url_for('login'))
    else:
        return render_template('home.html', user=user_data)

@app.route('/analysis')
def index():
    """Serves the main analysis page with upload form."""
    if session.get('user_data') is None:
        return redirect(url_for('login'))

    return render_template('index.html', conditions=CLASS_NAMES)

@app.route('/analyze', methods=['POST'])
def analyze():
    if DERSCAN_MODEL is None:
        return jsonify({'error': 'AI Model not loaded or available.'}), 500
    
    if 'skin_image' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    
    file = request.files['skin_image']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'No selected file or unsupported file type.'}), 400

    try:
        if file.filename is None:
             return jsonify({'error': 'Invalid file object received.'}), 400

        image_bytes = file.read()
        image_input = preprocess_image(io.BytesIO(image_bytes))
        
        predictions = DERSCAN_MODEL.predict(image_input)[0]
        
        results = []
        for i, class_name in enumerate(CLASS_NAMES):
            confidence_score = float(predictions[i]) * 100
            
            if confidence_score > 1.0: 
                 results.append({
                    'condition': class_name,
                    'confidence': f"{confidence_score:.2f}%" 
                })
        
        results.sort(key=lambda x: float(x['confidence'][:-1]), reverse=True)
        
        if not results:
            top_index = np.argmax(predictions)
            top_confidence = float(predictions[top_index]) * 100
            
            results.append({
                'condition': CLASS_NAMES[top_index] + ' (Top Prediction)',
                'confidence': f"{top_confidence:.2f}%",
                'is_critical': CLASS_NAMES[top_index] == 'Melanoma'
            })
            
        return jsonify({
            'success': True,
            'analysis': results,
            'message': 'Analysis complete. Review the detected conditions and confidence scores.'
        })

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return jsonify({'error': f'Internal Server Error during prediction: {e}'}), 500


if __name__ == '__main__':
    print("Ensure you run `python skin_model.py` first.")
    print("Starting Flask server...")
    app.run(debug=True)
