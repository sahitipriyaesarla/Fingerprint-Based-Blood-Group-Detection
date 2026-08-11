# Fingerprint Blood Group Detection

A deep learning project that predicts blood groups from fingerprint images using CNN.

## Project Structure

```
project123/
├── app.py                    # Flask web application
├── train_cnn_fast.py         # CNN training script
├── requirements.txt          # Python dependencies
├── dataset/                  # Training and test data
│   ├── train/               # Training images (8 blood groups)
│   └── test/                # Test images
├── utils/                   # Utility modules
│   ├── models/              # Trained model files
│   │   ├── blood_group_model.h5
│   │   └── class_indices.json
│   └── predict.py           # Prediction logic
├── static/                  # Static web assets
│   └── input_images/        # Uploaded images folder
└── templates/               # HTML templates
    ├── dashboard.html
    ├── index.html
    ├── probability.html
    └── static.html
```

## Blood Groups Supported

- A+, A-, AB+, AB-
- B+, B-, O+, O-

## Setup and Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model (optional - model already included):
   ```bash
   python train_cnn_fast.py
   ```

3. Run the web application:
   ```bash
   python app.py
   ```

4. Open browser and go to: `http://localhost:5000`

## Usage

1. Upload a fingerprint image
2. Click "Detect Blood Group"
3. View the prediction with confidence score
4. See probability distribution across all blood groups

## Model Performance

- CNN architecture with 4 convolutional blocks
- Trained on 6000+ fingerprint images
- Image preprocessing with CLAHE and adaptive thresholding
- Confidence-based predictions with probability distributions

## Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras
- **Image Processing**: OpenCV
- **Frontend**: HTML, CSS, JavaScript
