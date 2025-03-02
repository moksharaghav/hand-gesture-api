from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64
from flask_cors import CORS
import mediapipe as mp

# Load trained model
model = tf.keras.models.load_model("hand_gesture_model.h5")
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend to communicate with backend

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json["image"]
        image_data = base64.b64decode(data)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert image to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)

        # If no hand is detected, return "No hand detected"
        if not results.multi_hand_landmarks:
            return jsonify({"gesture": "No hand detected", "confidence": 0})

        # If hand is detected, crop and process image
        h, w, _ = img.shape
        for hand_landmarks in results.multi_hand_landmarks:
            x_min = min([int(landmark.x * w) for landmark in hand_landmarks.landmark])
            y_min = min([int(landmark.y * h) for landmark in hand_landmarks.landmark])
            x_max = max([int(landmark.x * w) for landmark in hand_landmarks.landmark])
            y_max = max([int(landmark.y * h) for landmark in hand_landmarks.landmark])

            # Crop hand region
            hand_img = img[y_min:y_max, x_min:x_max]

            # Preprocess image for model prediction
            hand_img = cv2.resize(hand_img, (128, 128))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            # Make prediction
            predictions = model.predict(hand_img)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions) * 100  # Convert to percentage

            return jsonify({"gesture": class_labels[predicted_class], "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)

