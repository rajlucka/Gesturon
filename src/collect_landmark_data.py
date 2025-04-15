# collect_landmark_data_append.py

import cv2
import numpy as np
import os
import mediapipe as mp
from pathlib import Path

# CONFIGURATION
DATA_PATH = os.path.join('data')  # Parent data folder
GESTURES = ['hello', 'thanks', 'yes', 'no']  # You can add more
NUM_NEW_SEQUENCES = 30  # Number of new sequences to add
SEQUENCE_LENGTH = 30  # Number of frames per sequence

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
        # Determine the next sequence number
        gesture_path = os.path.join(DATA_PATH, gesture)
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
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                # Draw landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
                else:
                    landmarks = np.zeros((21, 3))

                # Save landmarks
                dirpath = os.path.join(DATA_PATH, gesture, str(sequence))
                Path(dirpath).mkdir(parents=True, exist_ok=True)
                np.save(os.path.join(dirpath, f'{frame_num}.npy'), landmarks)

                # Overlay info
                cv2.putText(frame, f'{gesture.upper()} | Seq: {sequence} | Frame: {frame_num}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Collecting Landmarks', frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()