# continuous_recognition.py
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from collections import deque

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

def run_continuous_recognition():
    # Load model and encoder
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
        return
        
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Initialize sequences and predictions
    sequence = deque(maxlen=SEQ_LENGTH)
    sentence = []
    predictions = deque(maxlen=10)
    threshold = 0.7
    cooldown = 0

    # Webcam setup
    cap = cv2.VideoCapture(0)

    # Create a named window for trackbar controls
    cv2.namedWindow('Continuous Sign Recognition')

    # Create trackbars for adjustment
    cv2.createTrackbar('Contrast', 'Continuous Sign Recognition', 130, 300, lambda x: x)
    cv2.createTrackbar('Brightness', 'Continuous Sign Recognition', 10, 100, lambda x: x)
    cv2.createTrackbar('CLAHE Limit', 'Continuous Sign Recognition', 20, 50, lambda x: x)

    # Auto contrast adaptation variables
    auto_adapt = True
    target_brightness = 130  # Target mean brightness value
    target_contrast = 60     # Target standard deviation
    adaptation_rate = 0.05   # How quickly to adapt (0-1)

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
            
            # Calculate current frame metrics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_brightness = np.mean(gray)
            current_contrast = np.std(gray)
            
            # Get current trackbar values
            alpha = cv2.getTrackbarPos('Contrast', 'Continuous Sign Recognition') / 100.0
            beta = cv2.getTrackbarPos('Brightness', 'Continuous Sign Recognition')
            clip_limit = cv2.getTrackbarPos('CLAHE Limit', 'Continuous Sign Recognition') / 10.0
            
            # Auto-adapt parameters if enabled
            if auto_adapt:
                # Adjust contrast based on current contrast
                if current_contrast < target_contrast:
                    alpha += adaptation_rate
                elif current_contrast > target_contrast + 20:
                    alpha -= adaptation_rate
                
                # Adjust brightness based on current brightness
                if current_brightness < target_brightness:
                    beta += adaptation_rate * 10
                elif current_brightness > target_brightness + 20:
                    beta -= adaptation_rate * 10
                    
                # Keep values in reasonable ranges
                alpha = np.clip(alpha, 0.8, 3.0)
                beta = np.clip(beta, -50, 100)
                
                # Update trackbars
                cv2.setTrackbarPos('Contrast', 'Continuous Sign Recognition', int(alpha * 100))
                cv2.setTrackbarPos('Brightness', 'Continuous Sign Recognition', int(beta))
            
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
            
            # Process the enhanced frame
            rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Default zeros if no hand detected
            landmarks = np.zeros((21, 3))
            hand_detected = False

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    enhanced_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                hand_detected = True

            # Normalize landmarks (wrist-based normalization)
            landmarks = landmarks - landmarks[0]
            
            # Add to sequence
            sequence.append(landmarks.flatten())
            
            # Check if we have enough frames and hand is detected
            if len(sequence) == SEQ_LENGTH and hand_detected:
                try:
                    # Prepare input
                    input_data = np.expand_dims(np.array(sequence), axis=0)
                    
                    # Predict
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
                except Exception as e:
                    print(f"Error during prediction: {e}")
            
            if cooldown > 0:
                cooldown -= 1
            
            # Display sentence with nicer styling
            sentence_text = ' '.join(sentence)
            # Add a semi-transparent background for better readability
            overlay = enhanced_frame.copy()
            cv2.rectangle(overlay, (10, 10), (enhanced_frame.shape[1]-10, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, enhanced_frame, 0.3, 0, enhanced_frame)
            cv2.putText(enhanced_frame, sentence_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display current sign
            if hand_detected and len(predictions) == 10:
                most_common = max(set(predictions), key=list(predictions).count)
                confidence = list(predictions).count(most_common) / 10
                current_sign = label_encoder.inverse_transform([most_common])[0]
                
                # Add a semi-transparent background
                overlay = enhanced_frame.copy()
                cv2.rectangle(overlay, (10, enhanced_frame.shape[0]-80), (300, enhanced_frame.shape[0]-10), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, enhanced_frame, 0.3, 0, enhanced_frame)
                
                color = (0, 255, 0) if confidence > threshold else (0, 165, 255)
                cv2.putText(enhanced_frame, f"Current: {current_sign}", (20, enhanced_frame.shape[0]-50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                # Add confidence bar
                bar_length = int(confidence * 200)
                cv2.rectangle(enhanced_frame, (150, enhanced_frame.shape[0]-30), (150 + bar_length, enhanced_frame.shape[0]-20), color, -1)
                cv2.rectangle(enhanced_frame, (150, enhanced_frame.shape[0]-30), (350, enhanced_frame.shape[0]-20), (255, 255, 255), 1)
            else:
                # Show frames needed
                frames_needed = SEQ_LENGTH - len(sequence)
                if frames_needed > 0:
                    message = f"Collecting frames: {len(sequence)}/{SEQ_LENGTH}"
                    cv2.putText(enhanced_frame, message, (20, enhanced_frame.shape[0]-50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    # Show progress bar
                    progress = int((len(sequence) / SEQ_LENGTH) * 200)
                    cv2.rectangle(enhanced_frame, (150, enhanced_frame.shape[0]-30), (150 + progress, enhanced_frame.shape[0]-20), (0, 0, 255), -1)
                    cv2.rectangle(enhanced_frame, (150, enhanced_frame.shape[0]-30), (350, enhanced_frame.shape[0]-20), (255, 255, 255), 1)
                else:
                    cv2.putText(enhanced_frame, "No hand detected", (20, enhanced_frame.shape[0]-50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Display lighting metrics
            enhanced_gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
            enhanced_brightness = np.mean(enhanced_gray)
            enhanced_contrast = np.std(enhanced_gray)
            
            cv2.putText(enhanced_frame, f"Brightness: {enhanced_brightness:.1f}", (enhanced_frame.shape[1] - 200, enhanced_frame.shape[0] - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(enhanced_frame, f"Contrast: {enhanced_contrast:.1f}", (enhanced_frame.shape[1] - 200, enhanced_frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Auto-adapt status
            cv2.putText(enhanced_frame, f"Auto-adapt: {'ON' if auto_adapt else 'OFF'}", 
                       (enhanced_frame.shape[1] - 200, enhanced_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if auto_adapt else (0, 0, 255), 1)
            
            # Instructions
            cv2.putText(enhanced_frame, "Press 'q' to quit, 'c' to clear sentence, 'a' to toggle auto-adapt", 
                        (10, enhanced_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow('Continuous Sign Recognition', enhanced_frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence = []  # Clear the sentence
            elif key == ord('a'):
                auto_adapt = not auto_adapt  # Toggle auto-adapt

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_continuous_recognition()