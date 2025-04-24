Real-Time 3D Hand Pose-Based ASL Recognition
Author: Raj Vaibhav Lucka
Institution: Northeastern University
Course: Pattern Recognition and Computer Vision
Project Type: Computer Vision Final Project

Group Members
* Raj Vaibhav Lucka (Lead Developer & Presenter)

Project Description
This project presents a real-time American Sign Language (ASL) gesture recognition system built using computer vision and deep learning. The system leverages MediaPipe Hands for tracking 21 keypoints of the hand in 3D space and a compact Conv1D + LSTM + Attention deep learning model to classify gestures over time.
Key Capabilities:
* Works in real-time (~30 FPS) using only a webcam
* Handles challenging lighting using image enhancement + auto-adaptive contrast
* Recognizes 8 ASL gestures: hello, thanks, yes, no, sorry, where, i love you, and neutral
* Achieves 96.8% validation accuracy and 94.2% real-world accuracy

Presentation & Demo Videos
* Presentation Video: https://northeastern.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=b91c0348-616d-4c71-9b4f-b2c900700d87
* Live Demo Video: https://northeastern.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=aa2f7fb2-668d-4fc5-a4b7-b2c9005c23af&start=0

Technologies Used
* Python 3.x
* MediaPipe (Hands)
* OpenCV
* TensorFlow / Keras
* NumPy, Pandas
* CLAHE, Histogram Equalization
* Matplotlib / Seaborn (Evaluation)

System Architecture
The system follows a pipeline architecture with four main components:
1. Hand Tracking & Landmark Detection
Using MediaPipe's hand tracking solution to locate and track 21 3D landmarks on the hand at 30 FPS.
2. Preprocessing & Feature Extraction 
o Wrist-based normalization
o Scale normalization
o Sequence collection (30 frames)
o Image enhancement for lighting robustness
3. Deep Learning Model 
o Conv1D layer (64 filters) for spatial pattern extraction
o LSTM layer (128 units) for temporal dynamics
o Attention mechanism for focusing on key frames
o Dense layers for classification
4. Prediction Smoothing & Visualization 
o Temporal smoothing over 8 frames
o Confidence-based thresholding (0.6)
o Cooldown mechanism for stable output
o Real-time visualization with feedback

Results & Performance
Model Performance
* Training Accuracy: 99.8%
* Validation Accuracy: 96.8%
* Real-world Accuracy: 94.2%
* Average Response Time: 0.47 seconds
* Frame Rate: ~28-30 FPS on consumer hardware
Visualization
* t-SNE Feature Space: Clear clustering of gesture classes
* Confusion Matrix: Minimal confusion between gestures
* Learning Curves: Rapid convergence (10-15 epochs)
* Lighting Robustness: Maintains recognition in varied lighting conditions

Image Enhancement Features
The system includes advanced image processing techniques to maintain reliable hand tracking in challenging lighting environments:
1. Adaptive Histogram Equalization
Equalizes brightness distribution while preserving local details.
2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
Enhances local contrast without amplifying noise.
3. YUV Color Space Processing
Separates luminance from color for more effective enhancement.
4. Auto-Adaptive Parameters
Automatically adjusts contrast and brightness based on current lighting conditions.
5. Interactive Controls
User-adjustable parameters via trackbars for fine-tuning.

Getting Started
Prerequisites
Python 3.8+
OpenCV
TensorFlow 2.x
MediaPipe
NumPy
Matplotlib
Seaborn
pyttsx3 (for text-to-speech)
Installation
bash
# Create a virtual environment (optional but recommended)
python -m venv signlang-env
source signlang-env/bin/activate  # On Windows: signlang-env\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe tensorflow numpy matplotlib seaborn pyttsx3
Running the Project
Run each script individually in the following order:
1. capture_landmarks.py - Test hand tracking
2. collect_landmark_data.py - Collect training data
3. Download Dataset: https://github.com/rajlucka/Gesturon
4. train_model.py - Train the model
5. real_time_predictor.py - Run real-time prediction
6. continuous_recognition.py - Run continuous recognition
7. visualization.py - Generate visualizations

Usage Guide
Each script serves a specific purpose:
1. capture_landmarks.py
Visualize hand landmarks without recognition.
2. collect_landmark_data.py
Record new sequences for training or extending the gesture set.
3. train_model.py
Train the model with collected data (includes data augmentation).
4. real_time_predictor.py
Real-time gesture recognition with visual feedback.
5. continuous_recognition.py
Build sentences by recognizing multiple gestures in sequence.
6. visualization.py
Generate visualizations of model performance and features.

Data Collection Process
Our data collection experiments were meticulously structured to capture the nuanced movements of ASL signs across different users and environments. We developed a specialized capture interface that guided participants through a systematic protocol of performing each gesture 30 times in succession, with each recording lasting exactly 30 frames (approximately one second) to ensure temporal consistency. The interface displayed real-time feedback showing the MediaPipe hand tracking overlay alongside frame counts and sequence numbers, helping participants maintain consistent hand positions and timing. We experimented with various camera positions and lighting conditions, finding that diffuse, moderate lighting positioned at a 45-degree angle produced optimal landmark tracking with minimal occlusion issues. To ensure diversity in our dataset, we varied recording distances (0.5-1.0 meters from the camera), hand orientations (±15° from frontal view), and execution speeds (some sequences performed deliberately slower or faster), enabling the model to learn from realistic variations. We additionally tested different backdrop colors and discovered that neutral, non-reflective backgrounds significantly improved tracking reliability, particularly for depth estimation. Throughout collection, we continuously monitored tracking quality, immediately discarding sequences with more than 30% dropped frames or poor landmark confidence scores below 0.7, and instructing participants to repeat these recordings with adjusted positioning or movement speed.

Advanced Features
Data Augmentation
* Rotation augmentation: Random rotations (±15°)
* Scaling augmentation: Random size variations (0.9-1.1×)
* Shift augmentation: Random positional shifts (±5%)
* Noise augmentation: Gaussian noise addition
Attention Mechanism
* Dynamic focus on the most important frames in each sequence
* Weights each frame's contribution to the final classification
* Significantly improves accuracy for similar gestures
Lighting Adaptation
* Auto-adaptive contrast enhancement
* Real-time quality metrics display
* Interactive parameter adjustment
Continuous Recognition
* Gesture cooldown to prevent repetition
* Sentence building with word accumulation
* Confidence-based filtering

Future Improvements
* Extend to larger vocabulary of ASL signs
* Implement user-specific calibration for improved accuracy
* Add non-manual features (facial expressions)
* Develop mobile application version
* Incorporate transfer learning from larger sign language datasets
* Implement dynamic time warping for variable-speed gestures

Acknowledgements
* MediaPipe team for their hand tracking implementation
* TensorFlow and Keras development teams
* Northeastern University Pattern Recognition and Computer Vision course

Contact
For questions or collaboration opportunities, please contact:
* Raj Vaibhav Lucka - lucka.r@northeastern.edu
