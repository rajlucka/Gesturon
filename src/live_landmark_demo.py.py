# capture_landmarks.py

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set up the hand detection model
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the color from BGR (OpenCV default) to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        # Draw the hand annotations on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # Display the output
        cv2.imshow('Hand Landmarks', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()