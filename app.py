# app.py
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import cv2

from utils.predict import predict_single_softmax

app = Flask(__name__)

UPLOAD_FOLDER = "static/input_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/detect")
def detect():
    return render_template("index.html")

@app.route("/probability")
def probability():
    return render_template("probability.html")

@app.route("/submit-correction", methods=["POST"])
def submit_correction():
    try:
        data = request.get_json()
        
        img_hash = data.get("img_hash")
        predicted_group = data.get("predicted_group")
        correct_group = data.get("correct_group")
        
        if not all([img_hash, predicted_group, correct_group]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Add correction to learning system
        from utils.personalized_predictor import add_user_correction
        add_user_correction(img_hash, predicted_group, correct_group)
        
        return jsonify({"success": True, "message": "Correction saved successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/static")
def static_graph():
    return render_template("static.html")


# ---------------- UPLOAD ----------------
@app.route("/upload-files", methods=["POST"])
def upload_files():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "Unsupported file type"}), 400

    # Remove old images
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # Check image is readable
    img = cv2.imread(path)
    if img is None:
        os.remove(path)
        return jsonify({"error": "Invalid or corrupted image file"}), 400

    return jsonify({"success": True}), 200


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    files = os.listdir(UPLOAD_FOLDER)
    if len(files) != 1:
        return jsonify({"error": "Upload exactly one fingerprint image"}), 400

    img_path = os.path.join(UPLOAD_FOLDER, files[0])

    # Prediction handled fully by predict.py
    result = predict_single_softmax(img_path)

    if result["status"] == "error":
        return jsonify({
            "error": result["message"]
        }), 400

    return jsonify({
        "success": True,
        "predictions": [{
            "file": files[0],
            "label": result["label"],
            "confidence": result["confidence"]
        }],
        # 🔥 IMPORTANT FOR GRAPH
        "distribution": result["distribution"]
    }), 200


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
