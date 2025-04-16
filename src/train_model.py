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
GESTURES = np.array(['hello', 'thanks', 'yes', 'no', 'neutral'])  # includes 'neutral' class
SEQUENCE_LENGTH = 30

# --- LOAD AND NORMALIZE DATA ---
sequences = []
labels = []

for gesture in GESTURES:
    gesture_path = os.path.join(DATA_PATH, gesture)
    for sequence in os.listdir(gesture_path):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            frame_path = os.path.join(gesture_path, sequence, f'{frame_num}.npy')
            frame = np.load(frame_path)

            # Normalize by subtracting wrist (landmark 0)
            frame -= frame[0]  # shape: (21, 3)

            window.append(frame)
        sequences.append(window)
        labels.append(gesture)

X = np.array(sequences)  # Shape: (samples, 30, 21, 3)
X = X.reshape(X.shape[0], X.shape[1], -1)  # → (samples, 30, 63)

# --- ENCODE LABELS ---
le = LabelEncoder()
y = le.fit_transform(labels)
y = to_categorical(y)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
plt.show()