import numpy as np
import os
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


YOGA_POSES = ['Mountain Pose', 'Tree Pose', 'Warrior Pose', 'Downward Dog', 'Cobra Pose']


def preprocess_landmarks(landmarks_array):
    
    landmarks_array = np.array(landmarks_array)
    
    
    flattened = landmarks_array.flatten()
    

    distances = []
    for i in range(len(landmarks_array)):
        for j in range(i+1, len(landmarks_array)):
            # Calculate Euclidean distance between landmarks i and j
            dist = np.linalg.norm(landmarks_array[i][:2] - landmarks_array[j][:2])
            distances.append(dist)
    
    features = np.concatenate([flattened, np.array(distances)])
    
    return features

# Function to create a model for yoga pose classification
def create_yoga_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to load and preprocess data
def load_data(data_file):
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found")
        return None, None
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        X = []
        y = []
        
        for sample in data:
            pose_name = sample['pose']
            landmarks = sample['landmarks']
            
            # Get pose index
            pose_idx = YOGA_POSES.index(pose_name) if pose_name in YOGA_POSES else -1
            
            if pose_idx >= 0:
                # Preprocess landmarks
                features = preprocess_landmarks(landmarks)
                
                X.append(features)
                y.append(pose_idx)
        
        return np.array(X), np.array(y)
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# Function to train the model
def train_model(data_file, model_output_dir='models'):
    # Create output directory if it doesn't exist
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load and preprocess data
    X, y = load_data(data_file)
    
    if X is None or y is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Loaded {len(X)} samples")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save the scaler
    scaler_path = os.path.join(model_output_dir, 'yoga_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, len(YOGA_POSES))
    y_test_cat = to_categorical(y_test, len(YOGA_POSES))
    
    # Create model
    model = create_yoga_model(X_train.shape[1], len(YOGA_POSES))
    print("Model created with the following architecture:")
    model.summary()
    
    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        os.path.join(model_output_dir, 'yoga_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train model
    history = model.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test_cat),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test_cat)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, 'training_history.png'))
    plt.show()
    
    print(f"Model and scaler saved to {model_output_dir}")
    print(f"You can now use the trained model with the app.py script")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Yoga Pose Detection Model')
    parser.add_argument('--data', type=str, default='yoga_data/collected_poses.json',
                        help='Path to the data file')
    parser.add_argument('--output', type=str, default='models',
                        help='Directory to save the trained model')
    
    args = parser.parse_args()
    
    train_model(args.data, args.output)