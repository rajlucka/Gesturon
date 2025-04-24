# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model, Model
import pickle
import os
from sklearn.manifold import TSNE
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import cv2
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder

# Set up MediaPipe for hand visualization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def visualize_features():
    """Create t-SNE visualization of model features"""
    # Load model and data
    try:
        model = load_model('sign_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Extract sequence length from model
        SEQ_LENGTH = model.input_shape[1]
        print(f"Model loaded with input shape: {model.input_shape}, sequence length: {SEQ_LENGTH}")
    except FileNotFoundError:
        print("Error: Model or label encoder file not found. Train the model first.")
        return
        
    # Create a directory for visualizations if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Load training data with correct sequence length
    print("Loading data for visualization...")
    DATA_PATH = os.path.join('data')
    sequences = []
    labels = []
    
    # Load a sample of data from each class
    for gesture in os.listdir(DATA_PATH):
        gesture_path = os.path.join(DATA_PATH, gesture)
        if not os.path.isdir(gesture_path):
            continue
            
        count = 0
        for sequence in os.listdir(gesture_path):
            if count >= 5:  # Limit to 5 sequences per class
                break
                
            seq_path = os.path.join(gesture_path, sequence)
            if not os.path.isdir(seq_path):
                continue
                
            window = []
            has_frames = False
            
            for frame_num in range(SEQ_LENGTH):
                frame_path = os.path.join(seq_path, f'{frame_num}.npy')
                if not os.path.exists(frame_path):
                    frame = np.zeros((21, 3))
                else:
                    frame = np.load(frame_path)
                    has_frames = True
                    
                # Normalize
                frame = frame - frame[0]
                window.append(frame)
            
            if has_frames:
                sequences.append(window)
                labels.append(gesture)
                count += 1
    
    if not sequences:
        print("No data found to visualize!")
        return
        
    # Convert to numpy arrays
    X = np.array(sequences).reshape(-1, SEQ_LENGTH, 63)
    
    # Encode labels
    le_vis = LabelEncoder()
    y = le_vis.fit_transform(labels)
    
    print(f"Prepared visualization data with shape {X.shape}")
    
    # Create intermediate model to extract features
    input_layer = model.input
    
    # Find a good layer for feature extraction (not the output layer)
    feature_layer_name = None
    for layer in model.layers:
        if 'lstm' in layer.name or 'dense' in layer.name:
            if layer.name != model.layers[-1].name:  # Not the output layer
                feature_layer_name = layer.name
                break
    
    if feature_layer_name:
        print(f"Extracting features from layer: {feature_layer_name}")
        intermediate_model = Model(inputs=input_layer, 
                                  outputs=model.get_layer(feature_layer_name).output)
    else:
        # Fallback to the layer before output
        print("Using the layer before output for feature extraction")
        intermediate_model = Model(inputs=input_layer,
                                  outputs=model.layers[-2].output)
    
    # Get feature representations
    print("Extracting features...")
    features = intermediate_model.predict(X)
    if len(features.shape) == 3:  # If shape is (samples, timesteps, features)
        features = features[:, -1, :]  # Take the last timestep
    
    # Reduce dimensions for visualization using t-SNE
    print("Performing t-SNE dimensionality reduction...")
    perplexity = min(30, len(features)-1)  # Adjust perplexity based on data size
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(features)
    
    # Perform 3D t-SNE as well
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=perplexity)
    features_3d = tsne_3d.fit_transform(features)

    # Plot 2D t-SNE
    print("Generating 2D visualization...")
    plt.figure(figsize=(12, 10))
    
    # Use a distinct color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(y))))
    
    for i, label in enumerate(np.unique(y)):
        idx = y == label
        try:
            class_name = le_vis.inverse_transform([label])[0]
        except:
            class_name = f"Class {label}"
            
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                   color=colors[i], label=class_name, alpha=0.7)

    plt.legend()
    plt.title('t-SNE Visualization of Gesture Features')
    plt.savefig('visualizations/feature_visualization_2d.png')
    
    # Plot 3D t-SNE
    print("Generating 3D visualization...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, label in enumerate(np.unique(y)):
        idx = y == label
        try:
            class_name = le_vis.inverse_transform([label])[0]
        except:
            class_name = f"Class {label}"
            
        ax.scatter(features_3d[idx, 0], features_3d[idx, 1], features_3d[idx, 2],
                  color=colors[i], label=class_name, alpha=0.7)
    
    ax.legend()
    plt.title('3D t-SNE Visualization of Gesture Features')
    plt.savefig('visualizations/feature_visualization_3d.png')
    
    # Create prediction heatmap
    plt.figure(figsize=(12, 8))
    num_samples = min(20, len(X))
    
    # Make predictions
    preds = model.predict(X[:num_samples])
    
    # Get class names
    try:
        class_names = label_encoder.classes_
    except:
        class_names = [f"Class {i}" for i in range(preds.shape[1])]
    
    # Create heatmap
    sns.heatmap(preds, 
                xticklabels=class_names, 
                yticklabels=[f"Sample {i}" for i in range(num_samples)], 
                cmap="viridis", annot=True, fmt=".2f")
    plt.title("Prediction Probabilities for Test Samples")
    plt.tight_layout()
    plt.savefig('visualizations/prediction_heatmap.png')
    
    print("Visualizations saved to the 'visualizations' directory!")
    plt.show()

if __name__ == "__main__":
    visualize_features()