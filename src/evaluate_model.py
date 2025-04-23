# evaluate_model.py
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def evaluate_model():
    # Load model and test data
    try:
        model = load_model('sign_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        X_test_left = np.load('X_test_left.npy')
        X_test_right = np.load('X_test_right.npy')
        X_test_presence = np.load('X_test_presence.npy')
        X_test_features_pca = np.load('X_test_features_pca.npy')
        y_test = np.load('y_test.npy')
        print("Model and test data loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have trained the model and saved test data.")
        return
    
    # Create output directory
    os.makedirs('evaluation', exist_ok=True)
    
    # Make predictions
    print("Evaluating model on test data...")
    y_pred_probs = model.predict([X_test_left, X_test_right, X_test_presence, X_test_features_pca])
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = label_encoder.classes_
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('evaluation/confusion_matrix.png')
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open('evaluation/classification_report.txt', 'w') as f:
        f.write(report)
    print("\nClassification Report:")
    print(report)
    
    # Plot prediction probabilities for a few samples
    plt.figure(figsize=(12, 8))
    num_samples = min(20, len(X_test_left))
    sns.heatmap(y_pred_probs[:num_samples], 
                xticklabels=class_names,
                yticklabels=[f"Sample {i}" for i in range(num_samples)],
                cmap="viridis", annot=True, fmt=".2f")
    plt.title("Prediction Probabilities for Test Samples")
    plt.tight_layout()
    plt.savefig('evaluation/prediction_probabilities.png')
    
    # Plot per-class accuracy
    plt.figure(figsize=(10, 6))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    bar_plot = plt.bar(class_names, per_class_acc)
    plt.ylim([0, 1])
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45)
    
    # Add accuracy values on top of bars
    for idx, rect in enumerate(bar_plot):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig('evaluation/per_class_accuracy.png')
    
    # Analyze performance based on hand presence
    analyze_hand_presence_performance(X_test_presence, y_true, y_pred, class_names)
    
    print(f"\nEvaluation complete! Results saved to 'evaluation' directory.")
    plt.show()

def analyze_hand_presence_performance(presence_data, y_true, y_pred, class_names):
    """Analyze how model performance varies with hand presence patterns"""
    # Calculate average presence per sample
    avg_presence = np.mean(presence_data, axis=1)
    
    # Categories: left-only, right-only, both hands
    left_only = (avg_presence[:, 0] > 0.5) & (avg_presence[:, 1] < 0.5)
    right_only = (avg_presence[:, 0] < 0.5) & (avg_presence[:, 1] > 0.5)
    both_hands = (avg_presence[:, 0] > 0.5) & (avg_presence[:, 1] > 0.5)
    
    # Calculate accuracy for each category
    acc_left_only = np.mean(y_true[left_only] == y_pred[left_only]) if np.any(left_only) else 0
    acc_right_only = np.mean(y_true[right_only] == y_pred[right_only]) if np.any(right_only) else 0
    acc_both_hands = np.mean(y_true[both_hands] == y_pred[both_hands]) if np.any(both_hands) else 0
    
    # Plot results
    plt.figure(figsize=(10, 6))
    categories = ['Left Hand Only', 'Right Hand Only', 'Both Hands']
    accuracies = [acc_left_only, acc_right_only, acc_both_hands]
    counts = [np.sum(left_only), np.sum(right_only), np.sum(both_hands)]
    
    bar_plot = plt.bar(categories, accuracies)
    plt.ylim([0, 1])
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Hand Usage Pattern')
    
    # Add accuracy and count values on top of bars
    for idx, rect in enumerate(bar_plot):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height + 0.02,
                 f'{height:.2f} (n={counts[idx]})', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig('evaluation/hand_presence_accuracy.png')
    
    # Also analyze per-class hand presence patterns
    plt.figure(figsize=(12, 6))
    
    # For each class, calculate hand presence distribution
    class_presence = {}
    for i, class_idx in enumerate(np.unique(y_true)):
        mask = (y_true == class_idx)
        class_avg_presence = np.mean(avg_presence[mask], axis=0)
        class_presence[class_names[class_idx]] = class_avg_presence
    
    # Plot as stacked bars
    left_vals = [class_presence[name][0] for name in class_names]
    right_vals = [class_presence[name][1] for name in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, left_vals, width, label='Left Hand', color='green', alpha=0.7)
    ax.bar(x, right_vals, width, bottom=left_vals, label='Right Hand', color='blue', alpha=0.7)
    
    ax.set_ylabel('Average Hand Presence')
    ax.set_title('Hand Usage by Gesture Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('evaluation/class_hand_presence.png')

if __name__ == "__main__":
    evaluate_model()