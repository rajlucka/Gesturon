# run_demo.py
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='ASL Detection System')
    parser.add_argument('--mode', type=str, default='predict',
                      help='Mode to run: capture, collect, train, predict, continuous, visualize, evaluate, compare')
    
    args = parser.parse_args()
    
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths relative to the script directory
    file_paths = {
        'capture': os.path.join(script_dir, 'capture_landmarks.py'),
        'collect': os.path.join(script_dir, 'collect_landmark_data.py'),
        'train': os.path.join(script_dir, 'train_model.py'),
        'predict': os.path.join(script_dir, 'real_time_predictor.py'),
        'continuous': os.path.join(script_dir, 'continuous_recognition.py'),
        'visualize': os.path.join(script_dir, 'visualization.py'),
        'evaluate': os.path.join(script_dir, 'evaluate_model.py'),
        'compare': os.path.join(script_dir, 'model_comparison.py')
    }
    
    # Check if the selected file exists
    if args.mode in file_paths:
        file_path = file_paths[args.mode]
        if os.path.exists(file_path):
            print(f"Starting {args.mode} mode...")
            with open(file_path, 'r') as f:
                exec(f.read())
        else:
            print(f"Error: File {file_path} not found.")
            print(f"Have you created the {args.mode} script yet?")
            
            # Provide guidance for creating missing files
            if args.mode == 'predict' and not os.path.exists(file_paths['predict']):
                print("\nTo create real_time_predictor.py, you need to implement real-time ASL recognition.")
                print("The file should load your trained model and use your webcam for predictions.")
            
            if args.mode == 'continuous' and not os.path.exists(file_paths['continuous']):
                print("\nTo create continuous_recognition.py, you need to implement sentence building from ASL signs.")
                print("This is an advanced feature that builds on the basic real-time prediction.")
            
            if args.mode == 'visualize' and not os.path.exists(file_paths['visualize']):
                print("\nTo create visualization.py, you need to implement feature visualization tools.")
                print("This helps you understand how your model 'sees' the hand gestures.")
            
            if args.mode == 'evaluate' and not os.path.exists(file_paths['evaluate']):
                print("\nTo create evaluate_model.py, you need to implement detailed model evaluation metrics.")
                print("This will help you measure your model's performance.")
            
            if args.mode == 'compare' and not os.path.exists(file_paths['compare']):
                print("\nTo create model_comparison.py, you need to implement comparison of different model architectures.")
                print("This is useful for optimizing your model's performance.")
    else:
        print(f"Unknown mode: {args.mode}")
        print("Available modes: capture, collect, train, predict, continuous, visualize, evaluate, compare")
        
    print("\nASL Detection System complete.")

if __name__ == "__main__":
    main()