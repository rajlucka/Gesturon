# train_model.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Flatten, Attention, Add, GRU, BatchNormalization, Concatenate
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
DATA_PATH = os.path.join('data')
GESTURES = np.array(['hello', 'thanks', 'yes', 'no', 'neutral','my','name'])
SEQUENCE_LENGTH = 40

# --- IMPROVED FEATURE ENGINEERING ---
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

# --- AUGMENTATION FUNCTION ---
def augment_landmarks(landmarks, is_left=False, rotation_range=0.2, scale_range=0.1, shift_range=0.1):
    """Apply augmentation to hand landmarks"""
    augmented = landmarks.copy()
    
    # Skip if all zeros (no hand)
    if np.all(landmarks == 0):
        return augmented
    
    # Random rotation around wrist
    if np.random.random() < 0.5:
        angle = np.random.uniform(-rotation_range, rotation_range)
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        rotation_matrix = np.array([
            [cos_val, -sin_val, 0],
            [sin_val, cos_val, 0],
            [0, 0, 1]
        ])
        
        for i in range(len(augmented)):
            augmented[i] = augmented[i] @ rotation_matrix
    
    # Random scaling
    if np.random.random() < 0.5:
        scale_factor = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
        augmented *= scale_factor
    
    # Random shift
    if np.random.random() < 0.5:
        shift_x = np.random.uniform(-shift_range, shift_range)
        shift_y = np.random.uniform(-shift_range, shift_range)
        shift_z = np.random.uniform(-shift_range/2, shift_range/2)  # Less z-shifting
        
        shift = np.array([shift_x, shift_y, shift_z])
        for i in range(len(augmented)):
            augmented[i] += shift
    
    return augmented

# --- IMPROVED PREPROCESSING ---
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

# --- LOAD AND NORMALIZE DATA ---
sequences_left = []
sequences_right = []
hand_presence = []
advanced_features = []
labels = []

print("Loading and preprocessing data...")
for gesture in GESTURES:
    gesture_path = os.path.join(DATA_PATH, gesture)
    if not os.path.exists(gesture_path):
        print(f"Warning: Path {gesture_path} does not exist, skipping.")
        continue
        
    for sequence in os.listdir(gesture_path):
        seq_path = os.path.join(gesture_path, sequence)
        if not os.path.isdir(seq_path):
            continue
            
        window_left = []
        window_right = []
        window_presence = []
        sequence_features = []
        
        for frame_num in range(SEQUENCE_LENGTH):
            frame_path = os.path.join(seq_path, f'{frame_num}.npy')
            if not os.path.exists(frame_path):
                print(f"Warning: Frame {frame_path} does not exist, using zeros.")
                # Create empty data
                frame_data = {
                    'left_hand': np.zeros((21, 3)),
                    'right_hand': np.zeros((21, 3)),
                    'left_present': False,
                    'right_present': False
                }
            else:
                try:
                    frame_data = np.load(frame_path, allow_pickle=True).item()
                except:
                    print(f"Error loading {frame_path}, using zeros.")
                    # Create empty data
                    frame_data = {
                        'left_hand': np.zeros((21, 3)),
                        'right_hand': np.zeros((21, 3)),
                        'left_present': False,
                        'right_present': False
                    }
            
            # Get landmarks and presence flags
            left_hand = frame_data['left_hand']
            right_hand = frame_data['right_hand']
            left_present = frame_data['left_present']
            right_present = frame_data['right_present']
            
            # Apply preprocessing (note: preprocessing already applied during data collection)
            # But we'll apply it again to be safe
            if left_present:
                left_hand = preprocess_landmarks(left_hand, is_left_hand=True)
            if right_present:
                right_hand = preprocess_landmarks(right_hand, is_left_hand=False)
            
            # Extract advanced features
            frame_features = extract_advanced_features(
                right_hand, 
                landmarks_other=left_hand, 
                other_present=left_present
            )
            
            # Store presence as [left, right]
            presence = np.array([1.0 if left_present else 0.0, 1.0 if right_present else 0.0])
            
            window_left.append(left_hand)
            window_right.append(right_hand)
            window_presence.append(presence)
            sequence_features.append(frame_features)
            
        sequences_left.append(window_left)
        sequences_right.append(window_right)
        hand_presence.append(window_presence)
        advanced_features.append(sequence_features)
        labels.append(gesture)

X_left = np.array(sequences_left)  # Shape: (samples, sequence_length, 21, 3)
X_right = np.array(sequences_right)  # Shape: (samples, sequence_length, 21, 3)
X_presence = np.array(hand_presence)  # Shape: (samples, sequence_length, 2)

# Reshape landmark arrays
X_left = X_left.reshape(X_left.shape[0], X_left.shape[1], -1)  # → (samples, sequence_length, 63)
X_right = X_right.reshape(X_right.shape[0], X_right.shape[1], -1)  # → (samples, sequence_length, 63)

X_features = np.array(advanced_features)  # Shape: (samples, sequence_length, feature_count)

# --- ENCODE LABELS ---
le = LabelEncoder()
y = le.fit_transform(labels)
y = to_categorical(y)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# --- TRAIN/TEST SPLIT ---
X_train_left, X_test_left, X_train_right, X_test_right, X_train_presence, X_test_presence, X_train_features, X_test_features, y_train, y_test = train_test_split(
    X_left, X_right, X_presence, X_features, y, test_size=0.2, random_state=42)

# --- APPLY DATA AUGMENTATION ---
print("Applying data augmentation...")
augmented_left = []
augmented_right = []
augmented_presence = []
augmented_features = []
augmented_labels = []

# Add original data
augmented_left.extend(X_train_left)
augmented_right.extend(X_train_right)
augmented_presence.extend(X_train_presence)
augmented_features.extend(X_train_features)
augmented_labels.extend(y_train)

# Generate augmented samples
for i in range(len(X_train_left)):
    # Generate 2 augmented versions
    for _ in range(2):
        # Augment sequences
        aug_left_seq = []
        aug_right_seq = []
        aug_presence_seq = []
        aug_feat_seq = []
        
        for j in range(len(X_train_left[i])):
            # Extract original landmarks (reshape from flattened)
            original_left = X_train_left[i][j].reshape(21, 3)
            original_right = X_train_right[i][j].reshape(21, 3)
            original_presence = X_train_presence[i][j]
            
            # Apply augmentation
            augmented_left_hand = augment_landmarks(original_left, is_left=True)
            augmented_right_hand = augment_landmarks(original_right, is_left=False)
            
            # Extract features from augmented landmarks
            left_present = original_presence[0] > 0.5
            right_present = original_presence[1] > 0.5
            aug_features = extract_advanced_features(
                augmented_right_hand, 
                landmarks_other=augmented_left_hand, 
                other_present=left_present
            )
            
            # Save augmented data
            aug_left_seq.append(augmented_left_hand.flatten())
            aug_right_seq.append(augmented_right_hand.flatten())
            aug_presence_seq.append(original_presence)  # Keep original presence flags
            aug_feat_seq.append(aug_features)
        
        augmented_left.append(aug_left_seq)
        augmented_right.append(aug_right_seq)
        augmented_presence.append(aug_presence_seq)
        augmented_features.append(aug_feat_seq)
        augmented_labels.append(y_train[i])

# Convert to numpy arrays
X_train_left = np.array(augmented_left)
X_train_right = np.array(augmented_right)
X_train_presence = np.array(augmented_presence)
X_train_features = np.array(augmented_features)
y_train = np.array(augmented_labels)

# --- DIMENSIONALITY REDUCTION FOR FEATURES ---
print("Performing dimensionality reduction...")
# Reshape for PCA
X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features_flat = X_test_features.reshape(X_test_features.shape[0], -1)

# Apply PCA
n_components = min(100, X_train_features_flat.shape[1])  # Choose appropriate number
pca = PCA(n_components=n_components)
X_train_features_pca = pca.fit_transform(X_train_features_flat)
X_test_features_pca = pca.transform(X_test_features_flat)

# Save PCA model
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

# Save processed data for evaluation
np.save('X_test_left.npy', X_test_left)
np.save('X_test_right.npy', X_test_right)
np.save('X_test_presence.npy', X_test_presence)
np.save('X_test_features.npy', X_test_features)
np.save('X_test_features_pca.npy', X_test_features_pca)
np.save('y_test.npy', y_test)

# --- IMPROVED MODEL DEFINITION ---
def build_two_hand_model(left_shape, right_shape, presence_shape, features_pca_shape, num_classes):
    """Build model with separate inputs for left and right hands, presence flags, and extracted features"""
    # Left hand stream
    left_input = Input(shape=left_shape)
    x1 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(left_input)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = LSTM(128, return_sequences=True)(x1)
    x1 = Dropout(0.3)(x1)
    x1 = LSTM(64, return_sequences=False)(x1)
    
    # Right hand stream
    right_input = Input(shape=right_shape)
    x2 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(right_input)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    x2 = LSTM(128, return_sequences=True)(x2)
    x2 = Dropout(0.3)(x2)
    x2 = LSTM(64, return_sequences=False)(x2)
    
    # Hand presence stream
    presence_input = Input(shape=presence_shape)
    x3 = Conv1D(16, kernel_size=3, activation='relu', padding='same')(presence_input)
    x3 = Flatten()(x3)
    x3 = Dense(16, activation='relu')(x3)
    
    # Features stream (PCA reduced)
    features_input = Input(shape=(features_pca_shape,))
    x4 = Dense(128, activation='relu')(features_input)
    x4 = BatchNormalization()(x4)
    x4 = Dropout(0.3)(x4)
    x4 = Dense(64, activation='relu')(x4)
    
    # Combine all streams
    combined = Concatenate()([x1, x2, x3, x4])
    
    # Output layers
    x = Dense(256, activation='relu')(combined)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=[left_input, right_input, presence_input, features_input], outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# --- BUILD MODEL ---
print("Building and training model...")
left_shape = X_train_left.shape[1:]  # (sequence_length, 63)
right_shape = X_train_right.shape[1:]  # (sequence_length, 63)
presence_shape = X_train_presence.shape[1:]  # (sequence_length, 2)
features_pca_shape = X_train_features_pca.shape[1]  # PCA components

model = build_two_hand_model(left_shape, right_shape, presence_shape, features_pca_shape, len(GESTURES))
model.summary()

# --- TRAIN ---
history = model.fit(
    [X_train_left, X_train_right, X_train_presence, X_train_features_pca], 
    y_train, 
    epochs=80, 
    batch_size=8, 
    validation_data=([X_test_left, X_test_right, X_test_presence, X_test_features_pca], y_test)
)

# --- SAVE MODEL ---
model.save('sign_model.h5')
print("✅ Model saved as sign_model.h5")

# --- CONFUSION MATRIX ---
y_pred = model.predict([X_test_left, X_test_right, X_test_presence, X_test_features_pca])
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