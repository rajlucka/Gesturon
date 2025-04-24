# real_time_predictor.py

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import pyttsx3
from collections import deque

# Load model and label encoder
try:
    model = load_model('sign_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Automatically extract sequence length from model
    SEQ_LENGTH = model.input_shape[1]
    print(f"Model loaded successfully!")
    print(f"Input shape: {model.input_shape}, Sequence length: {SEQ_LENGTH}")
    print(f"Gestures: {label_encoder.classes_}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Text-to-speech engine
engine = pyttsx3.init()

# Initialize sequence with model's sequence length
sequence = deque(maxlen=SEQ_LENGTH)
prediction_threshold = 0.6
last_prediction = ''
cooldown_counter = 0

# Function for image enhancement
def enhance_image(image):
    # Convert to YUV color space
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # Apply histogram equalization to the Y channel
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    yuv[:,:,0] = clahe.apply(yuv[:,:,0])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # Apply additional contrast enhancement
    alpha = 1.3  # Contrast control (1.0 means no change)
    beta = 10    # Brightness control (0 means no change)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    return enhanced

# Prediction smoother class
class GestureSmoother:
    def __init__(self, num_classes, window_size=10, threshold=0.6):
        self.window_size = window_size
        self.threshold = threshold
        self.history = []
        self.num_classes = num_classes
        
    def update(self, prediction):
        """Add new prediction to history"""
        self.history.append(prediction)
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def get_smoothed_prediction(self):
        """Get smoothed prediction from history"""
        if not self.history:
            return None, 0
        
        # Average predictions over history
        avg_prediction = np.mean(self.history, axis=0)
        max_class = np.argmax(avg_prediction)
        confidence = avg_prediction[max_class]
        
        if confidence > self.threshold:
            return max_class, confidence
        return None, confidence

# Initialize the smoother
smoother = GestureSmoother(len(label_encoder.classes_), window_size=8, threshold=prediction_threshold)

# Webcam setup
cap = cv2.VideoCapture(0)

# Create a named window for trackbar controls
cv2.namedWindow('ASL Recognition')

# Create trackbars for adjustment
cv2.createTrackbar('Contrast', 'ASL Recognition', 130, 300, lambda x: x)
cv2.createTrackbar('Brightness', 'ASL Recognition', 10, 100, lambda x: x)
cv2.createTrackbar('CLAHE Limit', 'ASL Recognition', 20, 50, lambda x: x)

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
        
        # Get current trackbar values
        alpha = cv2.getTrackbarPos('Contrast', 'ASL Recognition') / 100.0
        beta = cv2.getTrackbarPos('Brightness', 'ASL Recognition')
        clip_limit = cv2.getTrackbarPos('CLAHE Limit', 'ASL Recognition') / 10.0
        
        # Apply custom image enhancement with trackbar values
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        
        # Apply CLAHE with adjustable clip limit
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        yuv[:,:,0] = clahe.apply(yuv[:,:,0])
        
        # Convert back to BGR
        enhanced_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # Apply contrast and brightness adjustment
        enhanced_frame = cv2.convertScaleAbs(enhanced_frame, alpha=alpha, beta=beta)
        
        # Create side-by-side view
        viz_frame = enhanced_frame.copy()
        
        # Calculate image quality metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced_gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness and contrast
        orig_brightness = np.mean(gray)
        enhanced_brightness = np.mean(enhanced_gray)
        
        orig_contrast = np.std(gray)
        enhanced_contrast = np.std(enhanced_gray)
        
        # Process the enhanced frame for hand detection
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Default zeros if no hand detected
        landmarks = np.zeros((21, 3))
        hand_present = False

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                viz_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            hand_present = True

        # Normalize landmarks (wrist-based normalization)
        landmarks = landmarks - landmarks[0]

        # Add to sequence
        sequence.append(landmarks.flatten())

        # Check if we have enough frames
        if len(sequence) == SEQ_LENGTH and hand_present:
            try:
                # Prepare input
                input_data = np.expand_dims(np.array(sequence), axis=0)
                
                # Get prediction
                prediction = model.predict(input_data)[0]
                
                # Update smoother
                smoother.update(prediction)
                smoothed_class, confidence = smoother.get_smoothed_prediction()
                
                if smoothed_class is not None:
                    predicted_label = label_encoder.inverse_transform([smoothed_class])[0]
                    
                    # Speak prediction (with cooldown)
                    if predicted_label != last_prediction and cooldown_counter == 0 and predicted_label != 'neutral':
                        last_prediction = predicted_label
                        print(f"Prediction: {predicted_label} ({confidence:.2f})")
                        engine.say(predicted_label)
                        engine.runAndWait()
                        cooldown_counter = 30
                    
                    if cooldown_counter > 0:
                        cooldown_counter -= 1
                    
                    # Display prediction
                    display_text = f"{predicted_label} ({confidence:.2f})"
                    cv2.putText(viz_frame, display_text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Show confidence bar
                    bar_width = int(confidence * 200)
                    cv2.rectangle(viz_frame, (10, 50), (10 + bar_width, 70), (0, 255, 0), -1)
                    cv2.rectangle(viz_frame, (10, 50), (210, 70), (255, 255, 255), 2)
                else:
                    cv2.putText(viz_frame, "Low confidence", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            except Exception as e:
                print(f"Error during prediction: {e}")
                cv2.putText(viz_frame, f"Error: {str(e)[:30]}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            # Show how many more frames we need
            frames_needed = SEQ_LENGTH - len(sequence)
            if frames_needed > 0:
                message = f"Collecting frames: {len(sequence)}/{SEQ_LENGTH}"
                cv2.putText(viz_frame, message, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show progress bar
                progress = int((len(sequence) / SEQ_LENGTH) * 200)
                cv2.rectangle(viz_frame, (10, 50), (10 + progress, 70), (0, 0, 255), -1)
                cv2.rectangle(viz_frame, (10, 50), (210, 70), (255, 255, 255), 2)
            else:
                cv2.putText(viz_frame, "No hand detected", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display image enhancement metrics
        cv2.putText(viz_frame, f"Original Brightness: {orig_brightness:.1f}", (10, viz_frame.shape[0] - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(viz_frame, f"Enhanced Brightness: {enhanced_brightness:.1f}", (10, viz_frame.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(viz_frame, f"Original Contrast: {orig_contrast:.1f}", (10, viz_frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(viz_frame, f"Enhanced Contrast: {enhanced_contrast:.1f}", (10, viz_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show quit instructions
        cv2.putText(viz_frame, "Press 'q' to quit", (viz_frame.shape[1] - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('ASL Recognition', viz_frame)
        
        # Check for quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()