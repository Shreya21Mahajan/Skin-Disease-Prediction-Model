AI-Based Skin Disease Prediction System
A Deep Learning Ensemble Model using EfficientNet & MobileNetV2 with Flask Web Deployment

AI-based Skin Disease Prediction Model that uses deep learning to address the worldwide shortage of dermatologists, especially in rural and underserved areas. 
This scarcity can lead to delayed diagnoses and negative health outcomes for patients. The project proposes a scalable and affordable solution that uses a Convolutional Neural Network (CNN) to analyze images of skin conditions. 
The system is designed to promptly identify common skin diseases such as melanoma, psoriasis, acne, and eczema.

Overview
This project is an AI-powered Skin Disease Classification System built using a Deep Learning Ensemble Model that combines EfficientNetB0 and MobileNetV2 for accurate and reliable predictions.
The system includes:
- A Flask web app
- A user-friendly interface for uploading skin images
- A trained ensemble model for prediction
- Secure login & registration pages
- Clean and modular ML code

Features
AI Model
- TensorFlow-based ensemble model
- Combines EfficientNet + MobileNetV2
- Trained on 9 classes of skin diseases
- Preprocessing with 224×224 image size

Web Application
- Built using Flask
- Upload image → get prediction instantly
- UI includes Home, Login, Register pages

Training
- Includes full training scripts
- Model saved in .keras format
- Data augmentation for better accuracy

Project Structure
app.py                                      # Flask application
ensemble_predict.py           # Ensemble prediction logic
skin_model.py                       # Loads models & constants
Efficientnet_train.py           # EfficientNet training script
Mobilenet_train.py              # MobileNet training script
templates/                             # HTML templates
requirements.txt                 # Environment dependencies
README.md                         # Documentation

Skin Disease Classes
i.	atopic_dermatitis
ii.	basal_cell_carcinoma
iii.	benign_keratosis-like_lesions
iv.	eczema
v.	melanocytic_nevi
vi.	melanoma
vii.	psoriasis_pictures_lichen_planus
viii.	seborrheic_keratoses
ix.	tinea_ringworm_candidiasis

How to Train the Model
Train EfficientNet: python Efficientnet_train.py
Train MobileNetV2: python Mobilenet_train.py
Run the Flask App
1. Create a virtual environment:
python -m venv venv
2. Activate environment:
Windows:
venv\Scripts\activate
Mac/Linux:
source venv/bin/activate
3. Install dependencies:
pip install -r requirements.txt
4. Run:
python app.py

Dataset
Dataset not included due to GitHub storage limits.
https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset
Dataset reduced to ~12K images and made the dataset balanced so it does show biasness for any one disease. Each class have images of ~2K images.

Installation
https://github.com/Shreya21Mahajan/Skin-Disease-Prediction-Model
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
Future Improvements
- Add Grad-CAM visualization
- Deploy to Render/HuggingFace Spaces
- Add dermatologist feedback module
- Implement more model architectures
