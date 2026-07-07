import os
import yaml
import tensorflow as tf

def load_config(config_path="config.yaml"):
    """Loads configuration parameters."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_dataset(data_dir, config, is_training=True):
    """Constructs an asynchronous tf.data.Dataset pipeline."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"CRITICAL: Directory {data_dir} does not exist. Run ingestion first.")

    img_size = (config['data']['img_size'], config['data']['img_size'])
    batch_size = config['data']['batch_size']
    
    # Ingest directly from disk structures mapped in Phase 2
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='binary',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=is_training,
        seed=42 if is_training else None
    )
    
    # Optimization constraints: CPU unblocking
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset

def get_augmentation_layer():
    """Compiles Keras preprocessing layers to combat extreme sample deficit (99 images)."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

if __name__ == "__main__":
    cfg = load_config()
    print("Compiling TensorFlow Input Pipeline...")
    
    train_ds = build_dataset(cfg['data']['train_dir'], cfg, is_training=True)
    val_ds = build_dataset(cfg['data']['val_dir'], cfg, is_training=False)
    
    print("\n=== Tensor Shape Verification ===")
    for image_batch, labels_batch in train_ds.take(1):
        print(f"Expected Input Geometry: (Batch Size, Height, Width, Channels)")
        print(f"Compiled Input Tensor:   {image_batch.shape}")
        print(f"Compiled Label Tensor:   {labels_batch.shape}")
        
    print("Pipeline architecture successfully compiled and ready for GPU.")