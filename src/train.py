import os
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.feature_engineering import load_dataset
from config import MODELS_DIR, MODEL_NAME

def train_model(X, y, out_path):
    # Train KNN
    print("Training K-Nearest Neighbors...")
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='cosine')
    knn.fit(X, y)
    
    # Save Model
    with open(out_path, "wb") as f:
        pickle.dump(knn, f)
    print(f"Model saved to {out_path}")

if __name__ == "__main__":
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        X, y, label_map = load_dataset()
        
        # Save label mapping for prediction
        label_map_path = os.path.join(MODELS_DIR, "label_map.pkl")
        with open(label_map_path, "wb") as f:
            pickle.dump(label_map, f)
            
        model_path = os.path.join(MODELS_DIR, MODEL_NAME)
        train_model(X, y, model_path)
    except Exception as e:
        print(f"Error during training: {e}")
