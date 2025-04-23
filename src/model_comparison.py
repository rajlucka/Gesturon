# model_comparison.py
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Flatten, Attention, Add, GRU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def compare_models():
    # Try to load preprocessed data
    try:
        X = np.load('X_preprocessed.npy')
        y = np.load('y_preprocessed.npy')
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        print("Preprocessed data not found. Please run train_model.py first.")
        return
    
    # Create output directory
    os.makedirs('model_comparison', exist_ok=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define model architectures
    models = {
        "Simple_LSTM": define_simple_lstm(X.shape[1:], y.shape[1]),
        "CNN_LSTM": define_cnn_lstm(X.shape[1:], y.shape[1]),
        "LSTM_Attention": define_lstm_attention(X.shape[1:], y.shape[1]),
        "GRU_Model": define_gru_model(X.shape[1:], y.shape[1]),
    }
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=8,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Measure time
        training_time = time.time() - start_time
        
        # Evaluate
        evaluation = model.evaluate(X_test, y_test, verbose=0)
        
        # Store results
        results[name] = {
            'history': history.history,
            'accuracy': evaluation[1],
            'loss': evaluation[0],
            'training_time': training_time
        }
        
        print(f"{name} - Accuracy: {evaluation[1]:.4f}, Loss: {evaluation[0]:.4f}, Time: {training_time:.2f}s")
    
    # Plot comparison results
    plt.figure(figsize=(10, 6))
    
    # Accuracy comparison
    accuracies = [results[model]['accuracy'] for model in models.keys()]
    plt.bar(models.keys(), accuracies, color='skyblue')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.xticks(rotation=45)
    
    # Add accuracy values on top of bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
        
    plt.tight_layout()
    plt.savefig('model_comparison/accuracy_comparison.png')
    
    # Training time comparison
    plt.figure(figsize=(10, 6))
    times = [results[model]['training_time'] for model in models.keys()]
    plt.bar(models.keys(), times, color='salmon')
    plt.title('Training Time Comparison')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(rotation=45)
    
    # Add time values on top of bars
    for i, v in enumerate(times):
        plt.text(i, v + 0.5, f"{v:.1f}s", ha='center')
        
    plt.tight_layout()
    plt.savefig('model_comparison/training_time_comparison.png')
    
    # Learning curves for each model
    plt.figure(figsize=(15, 10))
    for i, (name, result) in enumerate(results.items()):
        plt.subplot(2, 2, i+1)
        plt.plot(result['history']['accuracy'], label='Train')
        plt.plot(result['history']['val_accuracy'], label='Validation')
        plt.title(f'{name} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison/learning_curves.png')
    
    # Save best model
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    models[best_model].save('model_comparison/best_model.h5')
    
    print(f"\nComparison complete! The best model was {best_model} with accuracy {results[best_model]['accuracy']:.4f}")
    print(f"Results saved to 'model_comparison' directory.")

def define_simple_lstm(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=False, input_shape=input_shape),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def define_cnn_lstm(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def define_lstm_attention(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    attention_output = Attention(use_scale=True)([x, x])
    x = Add()([x, attention_output])
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def define_gru_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = GRU(128, return_sequences=True)(x)
    x = GRU(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    compare_models()