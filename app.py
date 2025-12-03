from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os

# Import unified prediction function
from skin_model import DERMACAN_ENSEMBLE, CLASS_NAMES, IMAGE_SIZE

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = "your_strong_secret_key_here"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# USER AUTH FUNCTIONS
DUMMY_USER_DATA = {
    "name": "Anil Sharma",
    "last_login": "2025-10-12 17:30 IST",
    "status": "Active",
    "notes": "Skin Analysis Check"
}

@app.route("/login", methods=["GET", "POST"])
def login():
    error = request.args.get("error")

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "Anil Sharma" and password == "sharmaji":
            session["user_data"] = DUMMY_USER_DATA
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid ID or Password.")

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user_data", None)
    return redirect(url_for("login"))


@app.route("/register")
def register():
    msg = request.args.get("status")
    return render_template("register.html", status_message=msg)


@app.route("/submit_request", methods=["POST"])
def submit_request():
    full_name = request.form.get("full_name")
    email = request.form.get("email")
    password = request.form.get("password")
    confirm = request.form.get("confirm_password")

    if password != confirm:
        return redirect(url_for("register", status="Error: Passwords do not match."))

    session["user_data"] = {
        "name": full_name,
        "last_login": "2025-10-12 18:08 IST",
        "status": "Active",
        "notes": "Wellness Check - Account Activated"
    }
    return redirect(url_for("home"))

# MAIN ROUTES
@app.route("/")
def home():
    user = session.get("user_data")
    if not user:
        return redirect(url_for("login"))
    return render_template("home.html", user=user)


@app.route("/analysis")
def analysis_page():
    user = session.get("user_data")
    if not user:
        return redirect(url_for("login"))
    return render_template("index.html", conditions=CLASS_NAMES)

# AI PREDICTION API
@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use JPG, JPEG, PNG."}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run ENSEMBLE prediction
        preds = DERMACAN_ENSEMBLE(img_array)

        predicted_index = int(np.argmax(preds[0]))
        confidence = float(preds[0][predicted_index])

        return jsonify({
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# START SERVER
if __name__ == "__main__":
    print("Starting DermaScan Flask server...")
    app.run(debug=True)
