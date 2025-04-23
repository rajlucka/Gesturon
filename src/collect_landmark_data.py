# collect_landmark_data.py

import cv2
import numpy as np
import os
import mediapipe as mp
from pathlib import Path

# CONFIGURATION
DATA_PATH = os.path.join('data')  # Parent data folder
GESTURES = ['hello', 'thanks', 'yes', 'no', 'neutral', 'my', 'name'] # You can add more
NUM_NEW_SEQUENCES = 30  # Number of new sequences to add
SEQUENCE_LENGTH = 40  # Number of frames per sequence

# Improved preprocessing function
def preprocess_landmarks(landmarks, is_left_hand=False):
    """Preprocess landmarks for better model performance"""
    # Skip if all zeros (no hand detected frame)
    if np.all(landmarks == 0):
        return landmarks
        
    # 1. Normalize by wrist position
    normalized = landmarks - landmarks[0]  # Subtract wrist position
    
    # 2. Scale normalization (make invariant to hand size)
    # Find distance between wrist and middle finger MCP (landmark 9)
    scale = np.linalg.norm(normalized[9])
    if scale > 0:  # Avoid division by zero
        normalized /= scale
    
    # 3. For left hand, mirror the x coordinates to standardize
    if is_left_hand:
        normalized[:, 0] = -normalized[:, 0]
    
    return normalized

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Start capturing
cap = cv2.VideoCapture(0)

# Set up Mediapipe Hands for two hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Track two hands
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

            # Start recording immediately without countdown
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                # Initialize arrays for both hands
                left_hand_landmarks = np.zeros((21, 3))
                right_hand_landmarks = np.zeros((21, 3))
                left_hand_present = False
                right_hand_present = False

                # Process hand landmarks if detected
                if results.multi_hand_landmarks:
                    # Create a copy of the frame for visualization
                    vis_frame = frame.copy()
                    
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        # Determine if this is a left or right hand
                        hand_label = handedness.classification[0].label
                        hand_color = (0, 255, 0) if hand_label == "Left" else (255, 0, 0)
                        
                        # Extract landmarks
                        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                        
                        # Store in appropriate array
                        if hand_label == "Left":
                            left_hand_landmarks = landmarks
                            left_hand_present = True
                        else:  # Right hand
                            right_hand_landmarks = landmarks
                            right_hand_present = True
                        
                        # Draw landmarks with styled connections on visualization frame
                        mp_drawing.draw_landmarks(
                            vis_frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Add label for hand type
                        wrist_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                        wrist_y = int(hand_landmarks.landmark[0].y * frame.shape[0])
                        cv2.putText(vis_frame, hand_label, (wrist_x, wrist_y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1, cv2.LINE_AA)
                    
                    # Use the visualization frame for display
                    frame = vis_frame

                # Apply preprocessing to landmarks
                processed_left = preprocess_landmarks(left_hand_landmarks, is_left_hand=True)
                processed_right = preprocess_landmarks(right_hand_landmarks, is_left_hand=False)
                
                # Combine both hands' landmarks and handedness flags
                combined_data = {
                    'left_hand': processed_left,
                    'right_hand': processed_right,
                    'left_present': left_hand_present,
                    'right_present': right_hand_present
                }

                # Save landmarks
                dirpath = os.path.join(DATA_PATH, gesture, str(sequence))
                Path(dirpath).mkdir(parents=True, exist_ok=True)
                np.save(os.path.join(dirpath, f'{frame_num}.npy'), combined_data)

                # Enhance overlay info with progress bars
                # Background rectangle for better visibility
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Text info
                cv2.putText(frame, f'{gesture.upper()} | Seq: {sequence} | Frame: {frame_num}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Progress bar for frame sequence
                progress = int((frame_num / SEQUENCE_LENGTH) * 200)
                cv2.rectangle(frame, (10, 40), (10 + progress, 50), (0, 255, 0), -1)
                cv2.rectangle(frame, (10, 40), (210, 50), (255, 255, 255), 1)
                
                # Hand presence indicators
                left_color = (0, 255, 0) if left_hand_present else (50, 50, 50)
                right_color = (0, 0, 255) if right_hand_present else (50, 50, 50)
                cv2.putText(frame, "L", (230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
                cv2.putText(frame, "R", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)
                
                cv2.imshow('Collecting Landmarks', frame)

                # Check for early exit
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()