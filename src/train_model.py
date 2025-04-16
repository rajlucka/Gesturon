# train_model.py (Upgraded)

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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

# Preprocess input
X = np.array(sequences)  # Shape: (samples, 30, 21, 3)
X = X.reshape(X.shape[0], X.shape[1], -1)  # Shape: (samples, 30, 63)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
y = to_categorical(y)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 63)))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(GESTURES), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=80, batch_size=8, validation_data=(X_test, y_test))

# Save model
model.save('sign_model.h5')
print("✅ Model saved as sign_model.h5")

# Evaluate with confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()