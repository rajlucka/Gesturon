# continuous_recognition.py
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from collections import deque

def preprocess_landmarks(landmarks):
    """Preprocess landmarks for better model performance"""
    # 1. Normalize by wrist position
    normalized = landmarks - landmarks[0]  # Subtract wrist position
    
    # 2. Scale normalization (make invariant to hand size)
    # Find distance between wrist and middle finger MCP (landmark 9)
    scale = np.linalg.norm(normalized[9])
    if scale > 0:  # Avoid division by zero
        normalized /= scale
    
    return normalized

def run_continuous_recognition():
    # Load model and encoder
    try:
        model = load_model('sign_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError:
        print("Error: Model or label encoder not found. Train the model first.")
        return
        
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Constants
    SEQUENCE_LENGTH = 40
    sequence = deque(maxlen=SEQUENCE_LENGTH)
    sentence = []
    predictions = deque(maxlen=10)
    threshold = 0.7
    cooldown = 0

    # Webcam setup
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

            landmarks = np.zeros((21, 3))
            hand_detected = False
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
                # Normalize
                landmarks = preprocess_landmarks(landmarks)
                hand_detected = True
                
                # Add to sequence
                sequence.append(landmarks)
                
                # Check if we have enough frames
                if len(sequence) == SEQUENCE_LENGTH:
                    # Predict
                    input_data = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 63)
                    res = model.predict(input_data)[0]
                    predictions.append(np.argmax(res))
                    
                    # Get most common prediction
                    if len(predictions) == 10 and cooldown == 0:
                        most_common = max(set(predictions), key=list(predictions).count)
                        confidence = list(predictions).count(most_common) / 10
                        
                        if confidence > threshold:
                            predicted_sign = label_encoder.inverse_transform([most_common])[0]
                            
                            # Only add to sentence if it's different from last prediction
                            if predicted_sign != 'neutral':
                                if len(sentence) == 0 or predicted_sign != sentence[-1]:
                                    sentence.append(predicted_sign)
                                    if len(sentence) > 5:  # Keep last 5 words
                                        sentence = sentence[-5:]
                                    cooldown = 20  # Set cooldown to avoid rapid predictions
            
            if cooldown > 0:
                cooldown -= 1
            
            # Display sentence
            sentence_text = ' '.join(sentence)
            cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, 50), (0, 0, 0), -1)
            cv2.putText(frame, sentence_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display current sign
            if hand_detected and len(predictions) == 10:
                most_common = max(set(predictions), key=list(predictions).count)
                confidence = list(predictions).count(most_common) / 10
                current_sign = label_encoder.inverse_transform([most_common])[0]
                
                cv2.rectangle(frame, (10, frame.shape[0]-60), (300, frame.shape[0]-10), (0, 0, 0), -1)
                cv2.putText(frame, f"Current: {current_sign}", (20, frame.shape[0]-35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                            (0, 255, 0) if confidence > threshold else (0, 165, 255), 2)
            
            # Instructions
            cv2.putText(frame, "Press 'q' to quit, 'c' to clear sentence", 
                        (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Continuous Sign Recognition', frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence = []  # Clear the sentence

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_continuous_recognition()