def extract_features(face_image):
    """Placeholder for feature extraction (embeddings, flattening)."""
    raise NotImplementedError

import os
import numpy as np
import cv2
from preprocessing import detect_and_crop_face, preprocess_face

# Constants
DATASET_PATH = "data/raw/"
IMAGE_SIZE = 64

def image_to_vector(image):
    """Flatten 64x64 grayscale image into 1D vector of length 4096"""
    if image is None:
        return None
    return image.flatten()


def load_dataset():
    """
    Main pipeline function:
    - Walks through data/raw/userX/ folders
    - Detects + crops face
    - Preprocesses
    - Flattens to vector
    - Builds X and y
    Returns: X (n_samples, 4096), y (n_samples,), label_to_user mapping
    """
    X = []
    y = []
    label_to_user = {}
    current_label = 0
    
    print("Starting dataset loading...\n")
    
    # Get user folders
    user_folders = [f for f in os.listdir(DATASET_PATH) 
                    if os.path.isdir(os.path.join(DATASET_PATH, f))]
    
    for user in sorted(user_folders):
        user_path = os.path.join(DATASET_PATH, user)
        label_to_user[current_label] = user
        
        image_count = 0
        skipped = 0
        
        for img_name in os.listdir(user_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(user_path, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Step 1: Detect and crop face
            cropped = detect_and_crop_face(image)
            if cropped is None:
                skipped += 1
                continue
            
            # Preprocess (resize, grayscale, normalize)
            processed = preprocess_face(cropped)
            if processed is None:
                skipped += 1
                continue
            
            # Flatten to vector
            vector = image_to_vector(processed)
            
            X.append(vector)
            y.append(current_label)
            image_count += 1
        
        print(f"Loaded {image_count} images for {user} (skipped {skipped} with no face detected)")
        current_label += 1
    
    if len(X) == 0:
        raise ValueError("No valid images were processed. Check your dataset and face detection.")
    
    X = np.array(X)
    y = np.array(y)
    
    print("\n=== Dataset Loaded Successfully ===")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Users: {len(label_to_user)}")
    print(f"Classes: {list(label_to_user.values())}")
    
    return X, y, label_to_user


# For quick testing when running the file directly
if __name__ == "__main__":
    try:
        X, y, label_map = load_dataset()
        print("\nFinal shapes:")
        print(f"X: {X.shape}")
        print(f"y: {y.shape}")
    except Exception as e:
        print(f"Error: {e}")