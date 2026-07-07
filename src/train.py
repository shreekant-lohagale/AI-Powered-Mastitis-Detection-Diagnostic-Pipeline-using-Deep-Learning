import os
import yaml
import tensorflow as tf
from pipeline import build_dataset, get_augmentation_layer

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_model(config):
    """Constructs the transfer learning architecture."""
    input_shape = (config['data']['img_size'], config['data']['img_size'], 3)
    
    # 1. Base Model Extraction
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=config['model']['weights']
    )
    
    # Freeze core feature extractors to prevent catastrophic forgetting
    base_model.trainable = False

    # 2. Custom Classification Head construction
    inputs = tf.keras.Input(shape=input_shape)
    
    # Inject Phase 3 Augmentation purely into the training graph
    x = get_augmentation_layer()(inputs)
    
    # Native MobileNetV2 expects [-1, 1] pixel scaling
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Aggressive dropout to penalize reliance on single features (Crucial for n=77)
    x = tf.keras.layers.Dropout(config['model']['dropout_rate'])(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['model']['learning_rate']),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

def main():
    cfg = load_config()
    
    print("Loading pipeline tensors...")
    train_ds = build_dataset(cfg['data']['train_dir'], cfg, is_training=True)
    val_ds = build_dataset(cfg['data']['val_dir'], cfg, is_training=False)
    
    print("Initializing MobileNetV2 Core...")
    model = build_model(cfg)
    model.summary()
    
    # Production Callbacks
    os.makedirs(os.path.dirname(cfg['logging']['checkpoint_dir']), exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=cfg['logging']['checkpoint_dir'],
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=cfg['logging']['log_dir'])
    ]
    
    print("\nInitiating Training Protocol...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg['model']['epochs'],
        callbacks=callbacks
    )
    print("\nTraining execution terminated. Optimal weights secured.")

if __name__ == "__main__":
    main()