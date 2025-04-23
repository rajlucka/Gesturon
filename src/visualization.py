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
    except FileNotFoundError:
        print("Error: Model or label encoder file not found. Train the model first.")
        return
        
    try:
        X_test_left = np.load('X_test_left.npy')
        X_test_right = np.load('X_test_right.npy')
        X_test_presence = np.load('X_test_presence.npy')
        X_test_features = np.load('X_test_features.npy')
        y_test = np.load('y_test.npy')
    except FileNotFoundError:
        print("Error: Test data files not found. Make sure to save them during training.")
        return

    # Create a directory for visualizations if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Create intermediate model to extract features from the right hand branch
    # (assuming right hand is primary for most signs)
    right_input = model.inputs[1]
    
    # Get the layer just before the flatten layer
    layer_name = None
    for layer in model.layers:
        if isinstance(layer, Model) or 'lstm' in layer.name:
            layer_name = layer.name
            break
    
    if layer_name:
        intermediate_model = Model(inputs=right_input, 
                                  outputs=model.get_layer(layer_name).output)
    else:
        # Fallback to the first LSTM layer
        for i, layer in enumerate(model.layers):
            if 'lstm' in layer.name:
                layer_name = layer.name
                break
        intermediate_model = Model(inputs=right_input, 
                                  outputs=model.get_layer(layer_name).output)
    
    # Get feature representations - take the last timestep only for sequence data
    print("Extracting features...")
    features = intermediate_model.predict(X_test_right)
    if len(features.shape) == 3:  # If shape is (samples, timesteps, features)
        features = features[:, -1, :]  # Take the last timestep
    
    # Reduce dimensions for visualization using t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    features_2d = tsne.fit_transform(features)
    
    # Perform 3D t-SNE as well
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=min(30, len(features)-1))
    features_3d = tsne_3d.fit_transform(features)

    # Plot 2D t-SNE
    print("Generating 2D visualization...")
    plt.figure(figsize=(12, 10))
    y_test_labels = np.argmax(y_test, axis=1)
    classes = label_encoder.classes_

    # Use a distinct color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    
    for i, label in enumerate(np.unique(y_test_labels)):
        idx = y_test_labels == label
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                   color=colors[i], label=classes[label], alpha=0.7)

    plt.legend()
    plt.title('t-SNE visualization of gesture features')
    plt.savefig('visualizations/feature_visualization_2d.png')
    
    # Plot 3D t-SNE
    print("Generating 3D visualization...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, label in enumerate(np.unique(y_test_labels)):
        idx = y_test_labels == label
        ax.scatter(features_3d[idx, 0], features_3d[idx, 1], features_3d[idx, 2],
                  color=colors[i], label=classes[label], alpha=0.7)
    
    ax.legend()
    plt.title('3D t-SNE visualization of gesture features')
    plt.savefig('visualizations/feature_visualization_3d.png')
    
    # Visualize hand presence patterns
    visualize_hand_presence(X_test_presence, y_test_labels, classes)
    
    # Visualize hand landmark connections and importance
    visualize_landmark_importance(X_test_left, X_test_right, X_test_presence, y_test_labels, classes)
    
    # Create feature heatmap visualization
    plt.figure(figsize=(12, 8))
    num_samples = min(20, len(X_test_right))
    
    # Sample predictions
    preds = model.predict([
        X_test_left[:num_samples],
        X_test_right[:num_samples],
        X_test_presence[:num_samples],
        X_test_features[:num_samples].reshape(num_samples, -1)[:, :model.inputs[3].shape[1]]
    ])
    
    # Create heatmap
    sns.heatmap(preds, 
                xticklabels=classes, 
                yticklabels=[f"Sample {i}" for i in range(num_samples)], 
                cmap="viridis", annot=True, fmt=".2f")
    plt.title("Prediction Probabilities for Test Samples")
    plt.tight_layout()
    plt.savefig('visualizations/prediction_heatmap.png')
    
    print("Visualizations saved to the 'visualizations' directory!")
    plt.show()

def visualize_hand_presence(presence_data, labels, class_names):
    """Visualize which hands are used for different gestures"""
    plt.figure(figsize=(12, 6))
    
    # For each class, calculate the average presence of left and right hands
    unique_labels = np.unique(labels)
    left_presence = []
    right_presence = []
    
    for label in unique_labels:
        class_idx = (labels == label)
        # Calculate mean presence across all frames in sequence
        mean_presence = np.mean(presence_data[class_idx], axis=(0, 1))
        left_presence.append(mean_presence[0])
        right_presence.append(mean_presence[1])
    
    # Plot as a grouped bar chart
    x = np.arange(len(unique_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, left_presence, width, label='Left Hand', color='green', alpha=0.7)
    ax.bar(x + width/2, right_presence, width, label='Right Hand', color='blue', alpha=0.7)
    
    ax.set_xlabel('Gesture')
    ax.set_ylabel('Average Hand Presence')
    ax.set_title('Hand Usage Patterns by Gesture')
    ax.set_xticks(x)
    ax.set_xticklabels([class_names[i] for i in unique_labels])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/hand_presence_patterns.png')

def visualize_landmark_importance(X_left, X_right, X_presence, labels, class_names):
    """Visualize which hand landmarks are most important for classification"""
    # Create directory for landmark visualizations
    os.makedirs('visualizations/landmarks', exist_ok=True)
    
    # Get all unique classes
    unique_classes = np.unique(labels)
    
    # For each class, visualize average hand pose
    for class_idx in unique_classes:
        # Get samples from this class
        class_samples_left = X_left[labels == class_idx]
        class_samples_right = X_right[labels == class_idx]
        class_samples_presence = X_presence[labels == class_idx]
        
        # Calculate average presence for this class
        avg_presence = np.mean(class_samples_presence, axis=(0, 1))
        left_present = avg_presence[0] > 0.5
        right_present = avg_presence[1] > 0.5
        
        # Create a black canvas
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Draw label at the top
        class_name = class_names[class_idx]
        cv2.putText(img, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Get hand presence info
        cv2.putText(img, f"Left hand: {'Present' if left_present else 'Not used'}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if left_present else (100, 100, 100), 2)
        cv2.putText(img, f"Right hand: {'Present' if right_present else 'Not used'}", (400, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0) if right_present else (100, 100, 100), 2)
        
        # Left hand visualization (if used)
        if left_present:
            # Average the landmarks across all samples and take the last frame
            avg_left = np.mean(class_samples_left, axis=0)[-1].reshape(21, 3)
            
            # Convert 3D landmarks to 2D for visualization
            landmarks_2d = np.zeros((21, 2), dtype=np.int32)
            landmarks_2d[:, 0] = (avg_left[:, 0] * -200 + 200).astype(np.int32)  # Mirror x-coord for left hand
            landmarks_2d[:, 1] = (avg_left[:, 1] * 200 + 200).astype(np.int32)
            
            # Draw connections
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = tuple(landmarks_2d[start_idx])
                end_point = tuple(landmarks_2d[end_idx])
                
                cv2.line(img, start_point, end_point, (0, 255, 0), 2)
            
            # Draw landmarks
            for i, point in enumerate(landmarks_2d):
                # Draw larger circles for fingertips
                if i in [4, 8, 12, 16, 20]:  # Fingertips
                    cv2.circle(img, tuple(point), 5, (0, 255, 255), -1)
                else:
                    cv2.circle(img, tuple(point), 3, (0, 255, 0), -1)
        
        # Right hand visualization (if used)
        if right_present:
            # Average the landmarks across all samples and take the last frame
            avg_right = np.mean(class_samples_right, axis=0)[-1].reshape(21, 3)
            
            # Convert 3D landmarks to 2D for visualization
            landmarks_2d = np.zeros((21, 2), dtype=np.int32)
            landmarks_2d[:, 0] = (avg_right[:, 0] * 200 + 600).astype(np.int32)
            landmarks_2d[:, 1] = (avg_right[:, 1] * 200 + 200).astype(np.int32)
            
            # Draw connections
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = tuple(landmarks_2d[start_idx])
                end_point = tuple(landmarks_2d[end_idx])
                
                cv2.line(img, start_point, end_point, (255, 0, 0), 2)
            
            # Draw landmarks
            for i, point in enumerate(landmarks_2d):
                # Draw larger circles for fingertips
                if i in [4, 8, 12, 16, 20]:  # Fingertips
                    cv2.circle(img, tuple(point), 5, (255, 255, 0), -1)
                else:
                    cv2.circle(img, tuple(point), 3, (255, 0, 0), -1)
        
        # Save the visualization
        cv2.imwrite(f'visualizations/landmarks/class_{class_name}_avg_pose.png', img)

    print("Landmark visualizations generated!")

if __name__ == "__main__":
    visualize_features()