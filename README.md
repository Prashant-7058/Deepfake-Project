## DeepFusion
DeepFusion - AI-Based Deepfake Detection
DeepFusion is a deep learning–powered web application that detects whether a given video is real or fake (deepfake). The system analyzes human facial features and temporal patterns using a hybrid CNN + LSTM model.

🚀 Features
🎥 Upload video and preview in-browser

🧠 Face detection using MediaPipe

🧬 Feature extraction using EfficientNetB4

🕒 Temporal sequence modeling using LSTM

🔍 Attention mechanism (optional) for enhanced detection

📊 Real/Fake result with confidence score

🌐 User-friendly web app interface (HTML/CSS + Flask backend)



🚀 Setup Instructions
# Create virtual environment
python -m venv venv

# Activate the environment
venv\Scripts\activate   # For Windows
# OR
source venv/bin/activate   # For macOS/Linux

# Install dependencies
pip install -r requirements.txt

# run 
python app.py


