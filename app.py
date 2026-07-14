from flask import Flask, request, jsonify, render_template
import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from timm import create_model
from deepfake_model import DeepFakeModel
import mediapipe as mp
import gc

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device("cpu")
print(device)

# Load EfficientNetB4 for feature extraction
efficientnet = None
model = None
face_detector = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_models():
    global efficientnet, model, face_detector

    if efficientnet is None:
        print("Loading EfficientNet...")

        efficientnet = create_model(
            "efficientnet_b4",
            pretrained=True
        )

        efficientnet = torch.nn.Sequential(
            *list(efficientnet.children())[:-2]
        )

        efficientnet.to(device)
        efficientnet.eval()

    if model is None:
        print("Loading LSTM...")

        model = DeepFakeModel().to(device)

        model.load_state_dict(
            torch.load(
                "models/cnn_lstm.pth",
                map_location=device
            )
        )

        model.eval()

    if face_detector is None:
        print("Loading MediaPipe...")

        mp_face_detection = mp.solutions.face_detection

        face_detector = mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )




def extract_features(video_path, target_frames=40):
    cap = cv2.VideoCapture(video_path)
    frames = []

    # fps = cap.get(cv2.CAP_PROP_FPS)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # duration_seconds = total_frames / fps if fps > 0 else 0

    # Fixed target frames to reduce memory usage
    target_frames = 40

    step = max(1, total_frames // target_frames)

    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)

            if results.detections:
                # Save the **original captured frame** (before crop)
                # frame_filename = os.path.join(save_folder, f"frame_{saved_frame_count:04d}.jpg")
                # cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1

                # Now crop the face and process it for feature extraction
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, w_box, h_box = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                                          int(bboxC.width * w), int(bboxC.height * h))
                    x, y = max(0, x), max(0, y)
                    w_box, h_box = min(frame.shape[1] - x, w_box), min(frame.shape[0] - y, h_box)

                    face = frame[y:y + h_box, x:x + w_box]
                    if face.shape[0] > 0 and face.shape[1] > 0:
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face = Image.fromarray(face)
                        face = transform(face).unsqueeze(0).to(device)

                        with torch.inference_mode():
                            feature = efficientnet(face)
                            feature = torch.mean(feature, dim=[2, 3]).cpu().numpy()[0]

                        del face
                        gc.collect()

                        frames.append(feature)
                        del feature
                        gc.collect()


        frame_count += 1

    cap.release()
    del cap
    gc.collect()

    video_features = np.array(frames)
    frames.clear()
    del frames
    gc.collect()
    if video_features.shape[0] > target_frames:
        indices = np.linspace(0, video_features.shape[0] - 1, target_frames, dtype=int)
        video_features = video_features[indices]
    elif video_features.shape[0] < target_frames:
        padding = np.zeros((target_frames - video_features.shape[0], 1792))
        video_features = np.vstack((video_features, padding))

    return video_features


def predict_deepfake(video_path):
    load_models()
    video_features = extract_features(video_path)

    features = torch.tensor(
        video_features,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    del video_features
    gc.collect()

    with torch.inference_mode():
        output = model(features)
        prob = torch.softmax(output, dim=1)
        confidence_score = torch.max(prob).item() * 100
        prediction = torch.argmax(prob, dim=1).item()

    del features
    del output
    del prob
    gc.collect()

    label = "Fake" if prediction == 1 else "Real"
    return label, round(confidence_score, 2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)

    try:
        label, confidence = predict_deepfake(video_path)
        # result = {
        #     "label": label,
        #     "confidence": confidence
        # }

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

        gc.collect()

    return jsonify({
        "label": label,
        "confidence": confidence
    })




if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        debug=False
    )
