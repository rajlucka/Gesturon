# real_time_predictor.py

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import pyttsx3
from collections import deque

# Load the trained model and label encoder
model = load_model('sign_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize Mediapipe and Text-to-Speech
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()

# Configuration
SEQUENCE_LENGTH = 30
sequence = deque(maxlen=SEQUENCE_LENGTH)
prediction_threshold = 0.6
last_prediction = ''
cooldown_counter = 0

# Start webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        landmarks = np.zeros((21, 3))  # Default if no hand detected

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

        sequence.append(landmarks)

        # Only predict if we have 30 frames
        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 63)
            prediction = model.predict(input_data)[0]
            max_confidence = np.max(prediction)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

            # Only speak if confidence is high and cooldown passed
            if max_confidence > prediction_threshold and predicted_label != last_prediction and cooldown_counter == 0:
                last_prediction = predicted_label
                print(f"🧠 Prediction: {predicted_label} ({max_confidence:.2f})")
                engine.say(predicted_label)
                engine.runAndWait()
                cooldown_counter = 30  # Skip next 30 frames to avoid repetition

            if cooldown_counter > 0:
                cooldown_counter -= 1

            # Show prediction on frame
            cv2.putText(frame, f'{predicted_label} ({max_confidence:.2f})', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Translator', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
