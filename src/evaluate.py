import os
import yaml
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from pipeline import build_dataset

def load_config(config_path="config.yaml"):
    """Loads centralized configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_pipeline():
    cfg = load_config()
    
    model_path = cfg['logging']['checkpoint_dir']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CRITICAL: Optimal weights not found at {model_path}")
        
    print(f"Loading compiled architecture and weights from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Constructing unshuffled validation tensor pipeline...")
    # is_training=False ensures shuffle=False, preserving the label order for strict 1:1 evaluation
    val_ds = build_dataset(cfg['data']['val_dir'], cfg, is_training=False)
    
    print("Extracting ground truth vectors...")
    y_true = np.concatenate([y for x, y in val_ds], axis=0)
    
    print("Executing batch inference...")
    y_pred_probs = model.predict(val_ds)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    print("\n=========================================")
    print("       ROBUST EVALUATION METRICS         ")
    print("=========================================\n")
    
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    print(f"True Negatives (Healthy predicted Healthy):  {cm[0][0]}")
    print(f"False Positives (Healthy predicted Mastitis): {cm[0][1]}")
    print(f"False Negatives (Mastitis predicted Healthy): {cm[1][0]}")
    print(f"True Positives (Mastitis predicted Mastitis): {cm[1][1]}\n")
    
    print("--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'Mastitis']))

if __name__ == "__main__":
    evaluate_pipeline()