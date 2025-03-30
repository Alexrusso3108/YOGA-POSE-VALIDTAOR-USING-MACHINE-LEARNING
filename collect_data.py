import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
import argparse

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define yoga poses to collect
YOGA_POSES = ['Mountain Pose', 'Tree Pose', 'Warrior Pose', 'Downward Dog', 'Cobra Pose']

# Function to extract landmarks
def extract_landmarks(results):
    landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return landmarks

# Function to save data
def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

# Function to collect data using webcam
def collect_data(output_dir='yoga_data', camera_id=0, samples_per_pose=30, delay=2):
    # Create data directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create window
    cv2.namedWindow('Yoga Pose Data Collection', cv2.WINDOW_NORMAL)
    
    all_data = []
    
    # Try to load existing data if available
    data_file = os.path.join(output_dir, 'collected_poses.json')
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                all_data = json.load(f)
            print(f"Loaded {len(all_data)} existing samples")
        except Exception as e:
            print(f"Error loading existing data: {e}")
    
    # Initialize pose detection
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        for pose_idx, pose_name in enumerate(YOGA_POSES):
            # Count existing samples for this pose
            existing_samples = sum(1 for sample in all_data if sample['pose'] == pose_name)
            samples_to_collect = max(0, samples_per_pose - existing_samples)
            
            if samples_to_collect <= 0:
                print(f"Already collected enough samples for {pose_name}, skipping")
                continue
            
            print(f"\nCollecting data for {pose_name} pose")
            print(f"Need to collect {samples_to_collect} more samples")
            print(f"Press 'c' to capture a sample, 's' to skip this pose, or 'q' to quit")
            
            samples_collected = 0
            last_capture_time = 0
            
            while samples_collected < samples_to_collect:
                # Read frame from webcam
                success, frame = cap.read()
                if not success:
                    print("Error: Failed to read frame from camera")
                    break
                
                # Flip the frame horizontally for a selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Convert the BGR image to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the image and detect pose
                results = pose.process(image_rgb)
                
                # Create a copy of the frame for drawing
                output_image = frame.copy()
                
                # Draw pose landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        output_image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Add text to the frame
                cv2.putText(output_image, f"Collecting: {pose_name}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(output_image, f"Samples: {samples_collected}/{samples_to_collect}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add instructions
                cv2.putText(output_image, "Press 'c' to capture", (10, output_image.shape[0] - 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(output_image, "Press 's' to skip pose", (10, output_image.shape[0] - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(output_image, "Press 'q' to quit", (10, output_image.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display the frame
                cv2.imshow('Yoga Pose Data Collection', output_image)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                
                # Capture sample
                current_time = time.time()
                if (key == ord('c') or (delay > 0 and current_time - last_capture_time >= delay)) and results.pose_landmarks:
                    # Extract landmarks
                    landmarks = extract_landmarks(results)
                    
                    if landmarks:
                        # Save the landmarks with the pose label
                        sample = {
                            'pose': pose_name,
                            'landmarks': landmarks,
                            'timestamp': time.time()
                        }
                        all_data.append(sample)
                        samples_collected += 1
                        last_capture_time = current_time
                        
                        print(f"  Sample {samples_collected}/{samples_to_collect} collected")
                        
                        # Save data after each sample
                        save_data(all_data, data_file)
                
                # Skip this pose
                elif key == ord('s'):
                    print(f"Skipping {pose_name} pose")
                    break
                
                # Quit
                elif key == ord('q'):
                    print("Data collection stopped by user")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Save all collected data
    save_data(all_data, data_file)
    print(f"\nData collection complete. Saved {len(all_data)} samples to {data_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect Yoga Pose Data')
    parser.add_argument('--output', type=str, default='yoga_data',
                        help='Directory to save collected data')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--samples', type=int, default=30,
                        help='Number of samples to collect per pose')
    parser.add_argument('--delay', type=int, default=0,
                        help='Automatic capture delay in seconds (0 to disable)')
    
    args = parser.parse_args()
    
    collect_data(args.output, args.camera, args.samples, args.delay)