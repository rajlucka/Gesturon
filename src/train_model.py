# train_model.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import pickle

# Config
DATA_PATH = os.path.join('data')
GESTURES = np.array(['hello', 'thanks', 'yes', 'no'])
SEQUENCE_LENGTH = 30

# Load sequences and labels
sequences = []
labels = []

for gesture in GESTURES:
    for sequence in os.listdir(os.path.join(DATA_PATH, gesture)):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            frame_path = os.path.join(DATA_PATH, gesture, sequence, f'{frame_num}.npy')
            frame = np.load(frame_path)
            window.append(frame)
        sequences.append(window)
        labels.append(gesture)

# Convert to numpy arrays
X = np.array(sequences)  # Shape: (num_samples, 30, 21, 3)
X = X.reshape(X.shape[0], X.shape[1], -1)  # Shape: (num_samples, 30, 63)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
y = to_categorical(y)  # One-hot encode

# Save the label encoder for future use
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 63)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(GESTURES), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save the model
model.save('sign_model.h5')
print("✅ Model trained and saved as 'sign_model.h5'")