import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("hand_gesture_model.h5")

# Define class labels (Update based on dataset)
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

IMG_SIZE = 128  # Must match training image size

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract bounding box coordinates
            h, w, _ = frame.shape
            x_min = min([int(landmark.x * w) for landmark in hand_landmarks.landmark])
            y_min = min([int(landmark.y * h) for landmark in hand_landmarks.landmark])
            x_max = max([int(landmark.x * w) for landmark in hand_landmarks.landmark])
            y_max = max([int(landmark.y * h) for landmark in hand_landmarks.landmark])

            # Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            # Preprocess for model prediction
            hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
            hand_img = hand_img / 255.0  # Normalize
            hand_img = np.expand_dims(hand_img, axis=0)  # Add batch dimension

            # Make prediction
            predictions = model.predict(hand_img)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions) * 100  # Convert to percentage

            # Display prediction only if confidence is high
            if confidence > 75:
                text = f"Gesture: {class_labels[predicted_class]} ({confidence:.2f}%)"
            else:
                text = "Gesture: Uncertain"

            # Show prediction on screen
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    else:
        # No hand detected → Show "No hand detected"
        cv2.putText(frame, "No hand detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
