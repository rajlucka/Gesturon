# train_model.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Flatten, Attention, Add
from tensorflow.keras.utils import to_categorical
import pickle

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data')

# Instead of hardcoding the gestures, detect them from the directory structure
available_gestures = []
for folder in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, folder)
    if os.path.isdir(folder_path):
        # Check if the folder has valid sequence data
        has_data = False
        for seq in os.listdir(folder_path):
            seq_path = os.path.join(folder_path, seq)
            if os.path.isdir(seq_path) and any(f.endswith('.npy') for f in os.listdir(seq_path)):
                has_data = True
                break
        if has_data:
            available_gestures.append(folder)

print(f"Detected gestures with data: {available_gestures}")
GESTURES = np.array(available_gestures)
SEQUENCE_LENGTH = 30

# --- DATA AUGMENTATION FUNCTIONS ---
def rotate_landmarks(landmarks, max_angle=15):
    """Apply random rotation to landmarks"""
    if np.all(landmarks == 0):  # Skip if no hand
        return landmarks
    
    # Convert to radians
    angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180
    
    # Create rotation matrix for x-y plane (2D rotation)
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    rotation_matrix = np.array([
        [cos_val, -sin_val, 0],
        [sin_val, cos_val, 0],
        [0, 0, 1]
    ])
    
    # Center around wrist
    centered = landmarks.copy()
    wrist = centered[0].copy()
    centered -= wrist
    
    # Apply rotation
    rotated = np.array([np.dot(point, rotation_matrix) for point in centered])
    
    # Shift back
    rotated += wrist
    
    return rotated

def scale_landmarks(landmarks, min_scale=0.9, max_scale=1.1):
    """Apply random scaling to landmarks"""
    if np.all(landmarks == 0):  # Skip if no hand
        return landmarks
    
    # Generate random scale factor
    scale = np.random.uniform(min_scale, max_scale)
    
    # Center around wrist
    centered = landmarks.copy()
    wrist = centered[0].copy()
    centered -= wrist
    
    # Apply scaling
    scaled = centered * scale
    
    # Shift back
    scaled += wrist
    
    return scaled

def shift_landmarks(landmarks, max_shift=0.05):
    """Apply random shift to landmarks"""
    if np.all(landmarks == 0):  # Skip if no hand
        return landmarks
    
    # Generate random shift
    shift_x = np.random.uniform(-max_shift, max_shift)
    shift_y = np.random.uniform(-max_shift, max_shift)
    shift_z = np.random.uniform(-max_shift/2, max_shift/2)  # Less z-axis shift
    
    shift = np.array([shift_x, shift_y, shift_z])
    
    # Apply shift
    shifted = landmarks + shift
    
    return shifted

def add_noise(landmarks, noise_level=0.005):
    """Add small random noise to landmarks"""
    if np.all(landmarks == 0):  # Skip if no hand
        return landmarks
    
    # Generate random noise
    noise = np.random.normal(0, noise_level, landmarks.shape)
    
    # Apply noise
    noisy = landmarks + noise
    
    return noisy

def augment_sequence(sequence):
    """Apply augmentation to a full sequence of landmarks"""
    augmented = []
    
    # Choose which augmentations to apply
    do_rotate = np.random.random() < 0.8
    do_scale = np.random.random() < 0.7
    do_shift = np.random.random() < 0.7
    do_noise = np.random.random() < 0.5
    
    for frame in sequence:
        frame_copy = frame.copy()
        
        # Apply chosen augmentations
        if do_rotate:
            frame_copy = rotate_landmarks(frame_copy)
        if do_scale:
            frame_copy = scale_landmarks(frame_copy)
        if do_shift:
            frame_copy = shift_landmarks(frame_copy)
        if do_noise:
            frame_copy = add_noise(frame_copy)
            
        augmented.append(frame_copy)
    
    return augmented

# --- LOAD AND NORMALIZE DATA ---
sequences = []
labels = []

print("Loading and preprocessing data...")
for gesture in GESTURES:
    gesture_path = os.path.join(DATA_PATH, gesture)
    if not os.path.exists(gesture_path):
        print(f"Warning: Path {gesture_path} does not exist, skipping.")
        continue
        
    # Count sequences with data
    valid_sequences = 0
    
    for sequence in os.listdir(gesture_path):
        seq_path = os.path.join(gesture_path, sequence)
        if not os.path.isdir(seq_path):
            continue
            
        window = []
        has_frames = False
        
        for frame_num in range(SEQUENCE_LENGTH):
            frame_path = os.path.join(seq_path, f'{frame_num}.npy')
            if not os.path.exists(frame_path):
                # print(f"Warning: Frame {frame_path} does not exist, using zeros.")
                frame = np.zeros((21, 3))
            else:
                frame = np.load(frame_path)
                has_frames = True

            # Normalize by wrist position
            frame = frame - frame[0]  # Subtract wrist position
            
            window.append(frame)
            
        # Only add sequences that have at least some valid frames
        if has_frames:
            sequences.append(window)
            labels.append(gesture)
            valid_sequences += 1
    
    print(f"Loaded {valid_sequences} valid sequences for gesture '{gesture}'")

if len(sequences) == 0:
    print("Error: No valid sequences found. Please collect data first.")
    exit(1)

# --- APPLY DATA AUGMENTATION ---
print("Applying data augmentation...")
augmented_sequences = []
augmented_labels = []

# Add original data
augmented_sequences.extend(sequences)
augmented_labels.extend(labels)

# Create augmented versions
for i, (sequence, label) in enumerate(zip(sequences, labels)):
    # Generate 2 augmented versions for each original sequence
    for _ in range(2):
        aug_sequence = augment_sequence(sequence)
        augmented_sequences.append(aug_sequence)
        augmented_labels.append(label)

print(f"Data augmentation complete. Total sequences: {len(augmented_sequences)} (including {len(sequences)} original)")

# Convert to numpy arrays
X = np.array(augmented_sequences)  # Shape: (samples, sequence_length, 21, 3)
X = X.reshape(X.shape[0], X.shape[1], -1)  # → (samples, sequence_length, 63)

# --- ENCODE LABELS ---
le = LabelEncoder()
y = le.fit_transform(augmented_labels)
y = to_categorical(y)

print(f"Training with {len(GESTURES)} gestures: {', '.join(GESTURES)}")
print(f"Input shape: {X.shape}, Label shape: {y.shape}")

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test data for later visualization
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# --- MODEL DEFINITION (Conv1D + LSTM + Attention) ---
input_shape = (SEQUENCE_LENGTH, 63)
inputs = Input(shape=input_shape)

x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
x = Dropout(0.3)(x)

x = LSTM(128, return_sequences=True)(x)
x = Dropout(0.3)(x)

# Self-attention over time
attention_output = Attention(use_scale=True)([x, x])
x = Add()([x, attention_output])

x = Flatten()(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(len(GESTURES), activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"Model output shape: {model.output_shape}")
model.summary()

# --- TRAIN ---
history = model.fit(X_train, y_train, epochs=80, batch_size=8, validation_data=(X_test, y_test))

# --- SAVE MODEL ---
model.save('sign_model.h5')
print("✅ Model saved as sign_model.h5")

# --- CONFUSION MATRIX ---
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')
plt.show()

# --- LEARNING CURVES ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('learning_curves.png')
plt.show()

print("Training complete!")