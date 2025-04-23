# continuous_recognition.py
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from collections import deque

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points in 3D space"""
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    # Avoid division by zero
    if v1_norm == 0 or v2_norm == 0:
        return 0
        
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    # Calculate angle using dot product
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot_product)
    
    return angle

def extract_advanced_features(landmarks, landmarks_other=None, other_present=False):
    """Extract advanced features from single frame landmarks"""
    # Skip if all zeros (no hand detected frame)
    if np.all(landmarks == 0):
        return np.zeros(53 + (53 if other_present else 0) + (10 if other_present else 0))
    
    # 1. Calculate finger angles
    # Thumb angle
    thumb_angle = calculate_angle(landmarks[1], landmarks[2], landmarks[3])
    # Index finger angle
    index_angle = calculate_angle(landmarks[5], landmarks[6], landmarks[7])
    # Middle finger angle
    middle_angle = calculate_angle(landmarks[9], landmarks[10], landmarks[11])
    # Ring finger angle
    ring_angle = calculate_angle(landmarks[13], landmarks[14], landmarks[15])
    # Pinky angle
    pinky_angle = calculate_angle(landmarks[17], landmarks[18], landmarks[19])
    
    angles = np.array([thumb_angle, index_angle, middle_angle, ring_angle, pinky_angle])
    
    # 2. Calculate distances between fingertips and wrist
    fingertip_indices = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
    distances = np.array([np.linalg.norm(landmarks[idx] - landmarks[0]) for idx in fingertip_indices])
    
    # 3. Calculate fingertip to fingertip distances (10 pairs)
    fingertip_distances = []
    for i in range(len(fingertip_indices)):
        for j in range(i+1, len(fingertip_indices)):
            fingertip_distances.append(
                np.linalg.norm(landmarks[fingertip_indices[i]] - landmarks[fingertip_indices[j]])
            )
    
    # 4. Hand shape features: convex hull area approximation
    # Use fingertips and palm landmarks
    key_points = np.array([landmarks[0], landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]])
    # Approximate area using distances
    shape_features = []
    for i in range(len(key_points)):
        for j in range(i+1, len(key_points)):
            shape_features.append(np.linalg.norm(key_points[i] - key_points[j]))
    
    # Combine single hand features
    single_hand_features = np.concatenate([angles, distances, fingertip_distances, shape_features])
    
    # If we have a second hand, compute inter-hand features
    if other_present and landmarks_other is not None and not np.all(landmarks_other == 0):
        # Extract features for other hand
        other_hand_features = extract_advanced_features(landmarks_other)
        
        # Calculate distances between key points on both hands
        inter_hand_features = []
        for idx1 in fingertip_indices + [0]:  # Add wrist
            for idx2 in fingertip_indices + [0]:  # Add wrist
                # Only use a subset of combinations to avoid too many features
                if idx1 == 0 or idx2 == 0 or (idx1 in [4, 8, 20] and idx2 in [4, 8, 20]):
                    inter_hand_features.append(np.linalg.norm(landmarks[idx1] - landmarks_other[idx2]))
        
        # Return combined features
        return np.concatenate([single_hand_features, other_hand_features, inter_hand_features])
    
    # Return single hand features only
    return single_hand_features

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

def run_continuous_recognition():
    # Load model and encoder
    try:
        model = load_model('sign_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        with open('pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)
        print("Model and resources loaded successfully!")
    except Exception as e:
        print(f"Error loading model or resources: {e}")
        print("Please make sure you've trained the model first.")
        return
        
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Constants
    SEQUENCE_LENGTH = 40
    sequence_left = deque(maxlen=SEQUENCE_LENGTH)
    sequence_right = deque(maxlen=SEQUENCE_LENGTH)
    sequence_presence = deque(maxlen=SEQUENCE_LENGTH)
    sentence = []
    predictions = deque(maxlen=10)
    threshold = 0.7
    cooldown = 0

    # Webcam setup
    cap = cv2.VideoCapture(0)
    
    # Initialization period to prevent false "no" detections
    initialization_frames = 30
    frame_count = 0
    initialized = False

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Track two hands
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

            # Initialize arrays for both hands
            left_hand_landmarks = np.zeros((21, 3))
            right_hand_landmarks = np.zeros((21, 3))
            left_hand_present = False
            right_hand_present = False

            # Process hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Determine if this is a left or right hand
                    hand_label = handedness.classification[0].label
                    hand_color = (0, 255, 0) if hand_label == "Left" else (255, 0, 0)
                    
                    # Extract landmarks
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    
                    # Store in appropriate array
                    if hand_label == "Left":
                        left_hand_landmarks = preprocess_landmarks(landmarks, is_left_hand=True)
                        left_hand_present = True
                    else:  # Right hand
                        right_hand_landmarks = preprocess_landmarks(landmarks, is_left_hand=False)
                        right_hand_present = True
                    
                    # Draw landmarks with styled connections
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Add label for hand type
                    wrist_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                    wrist_y = int(hand_landmarks.landmark[0].y * frame.shape[0])
                    cv2.putText(frame, hand_label, (wrist_x, wrist_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1, cv2.LINE_AA)
                
                # Update initialization counter
                if not initialized:
                    frame_count += 1
                    if frame_count >= initialization_frames:
                        initialized = True
            
            # Store presence as [left, right]
            presence = np.array([1.0 if left_hand_present else 0.0, 1.0 if right_hand_present else 0.0])
            
            # Add to sequences
            sequence_left.append(left_hand_landmarks.flatten())
            sequence_right.append(right_hand_landmarks.flatten())
            sequence_presence.append(presence)
            
            # Check if we have enough frames and initialization is complete
            if len(sequence_left) == SEQUENCE_LENGTH and initialized:
                # Extract features
                sequence_features = []
                for i in range(SEQUENCE_LENGTH):
                    left = sequence_left[i].reshape(21, 3)
                    right = sequence_right[i].reshape(21, 3)
                    left_present = sequence_presence[i][0] > 0.5
                    right_present = sequence_presence[i][1] > 0.5
                    
                    # Extract advanced features
                    frame_features = extract_advanced_features(
                        right, landmarks_other=left, other_present=left_present
                    )
                    sequence_features.append(frame_features)
                
                # Reshape and apply PCA
                features_flat = np.array(sequence_features).reshape(1, -1)
                features_pca = pca.transform(features_flat)
                
                # Prepare inputs
                input_left = np.array(sequence_left).reshape(1, SEQUENCE_LENGTH, 63)
                input_right = np.array(sequence_right).reshape(1, SEQUENCE_LENGTH, 63)
                input_presence = np.array(sequence_presence).reshape(1, SEQUENCE_LENGTH, 2)
                
                # Predict
                res = model.predict([input_left, input_right, input_presence, features_pca])[0]
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
            
            # Display sentence with nicer styling
            sentence_text = ' '.join(sentence)
            # Add a semi-transparent background for better readability
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (frame.shape[1]-10, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, sentence_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display current sign
            if (left_hand_present or right_hand_present) and len(predictions) == 10 and initialized:
                most_common = max(set(predictions), key=list(predictions).count)
                confidence = list(predictions).count(most_common) / 10
                current_sign = label_encoder.inverse_transform([most_common])[0]
                
                # Add a semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, frame.shape[0]-80), (300, frame.shape[0]-10), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                color = (0, 255, 0) if confidence > threshold else (0, 165, 255)
                cv2.putText(frame, f"Current: {current_sign}", (20, frame.shape[0]-50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                # Add confidence bar
                bar_length = int(confidence * 200)
                cv2.rectangle(frame, (150, frame.shape[0]-30), (150 + bar_length, frame.shape[0]-20), color, -1)
                cv2.rectangle(frame, (150, frame.shape[0]-30), (350, frame.shape[0]-20), (255, 255, 255), 1)
                
                # Hand presence indicators
                left_color = (0, 255, 0) if left_hand_present else (50, 50, 50)
                right_color = (255, 0, 0) if right_hand_present else (50, 50, 50)
                cv2.circle(frame, (30, frame.shape[0]-50), 8, left_color, -1)
                cv2.circle(frame, (100, frame.shape[0]-50), 8, right_color, -1)
            else:
                # Show initialization status
                if not initialized:
                    status = f"Initializing... {frame_count}/{initialization_frames}"
                    cv2.putText(frame, status, (20, frame.shape[0]-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Waiting for hand...", (20, frame.shape[0]-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Instructions
            cv2.putText(frame, "Press 'q' to quit, 'c' to clear sentence", 
                        (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
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