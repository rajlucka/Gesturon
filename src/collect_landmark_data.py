# collect_landmark_data.py

import cv2
import numpy as np
import os
import mediapipe as mp
from pathlib import Path

# CONFIGURATION
DATA_PATH = os.path.join('data')  # Parent data folder
GESTURES = ['hello', 'thanks', 'yes', 'no', 'neutral', 'sorry', 'where', 'i love you']
NUM_NEW_SEQUENCES = 30  # Number of new sequences to add
SEQUENCE_LENGTH = 30  # Number of frames per sequence

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

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start capturing
cap = cv2.VideoCapture(0)

# Set up Mediapipe Hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    for gesture in GESTURES:
        # Create gesture directory if it doesn't exist
        gesture_path = os.path.join(DATA_PATH, gesture)
        Path(gesture_path).mkdir(parents=True, exist_ok=True)
        
        # Determine the next sequence number
        existing_sequences = [
            int(seq) for seq in os.listdir(gesture_path)
            if os.path.isdir(os.path.join(gesture_path, seq)) and seq.isdigit()
        ] if os.path.exists(gesture_path) else []

        start_sequence = max(existing_sequences) + 1 if existing_sequences else 0

        print(f"\nStarting new data collection for gesture: {gesture}")
        print(f"Existing sequences found: {len(existing_sequences)}")
        print(f"Starting from sequence index: {start_sequence}")
        input("Press ENTER when you're ready to start recording...")

        for sequence in range(start_sequence, start_sequence + NUM_NEW_SEQUENCES):
            print(f"\nReady to record sequence {sequence} for '{gesture}'")
            print("Press ENTER to record this sequence, or press 's' and ENTER to skip this gesture.")

            user_input = input().strip().lower()
            if user_input == 's':
                print(f"🔁 Skipping remaining sequences for gesture '{gesture}'.")
                break

            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                
                # Apply image enhancement
                enhanced_frame = enhance_image(frame)
                
                # Process the enhanced frame
                rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                # Draw landmarks
                landmarks = np.zeros((21, 3))
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            enhanced_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])

                # Save landmarks
                dirpath = os.path.join(DATA_PATH, gesture, str(sequence))
                Path(dirpath).mkdir(parents=True, exist_ok=True)
                np.save(os.path.join(dirpath, f'{frame_num}.npy'), landmarks)

                # Overlay info on enhanced frame
                overlay = enhanced_frame.copy()
                cv2.rectangle(overlay, (0, 0), (enhanced_frame.shape[1], 60), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, enhanced_frame, 0.3, 0, enhanced_frame)
                
                cv2.putText(enhanced_frame, f'{gesture.upper()} | Seq: {sequence} | Frame: {frame_num}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Progress bar for frame sequence
                progress = int((frame_num / SEQUENCE_LENGTH) * 200)
                cv2.rectangle(enhanced_frame, (10, 40), (10 + progress, 50), (0, 255, 0), -1)
                cv2.rectangle(enhanced_frame, (10, 40), (210, 50), (255, 255, 255), 1)
                
                # Show lighting quality indicator
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray)
                std_brightness = np.std(gray)
                
                lighting_quality = "Good"
                lighting_color = (0, 255, 0)
                if avg_brightness < 80 or avg_brightness > 200:
                    lighting_quality = "Poor"
                    lighting_color = (0, 0, 255)
                elif std_brightness < 40:
                    lighting_quality = "Fair"
                    lighting_color = (0, 255, 255)
                
                cv2.putText(enhanced_frame, f"Lighting: {lighting_quality}", (10, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, lighting_color, 2)
                
                cv2.imshow('Collecting Landmarks', enhanced_frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()