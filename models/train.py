"""
Training Module
Handles model training loops, callbacks configuration, and history persistence.
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
from .config_models import (
    EARLY_STOPPING_PATIENCE, 
    REDUCE_LR_PATIENCE, 
    REDUCE_LR_FACTOR, 
    MIN_LR,
    DEFAULT_EPOCHS
)

def get_callbacks(model_name):
    """Creates validation callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint."""
    os.makedirs('weights', exist_ok=True)
    
    checkpoint_path = f'weights/{model_name}_best_model.h5'
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LR,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    return [early_stopping, reduce_lr, checkpoint]

def train_single_model(model, model_name, train_dataset, val_dataset, epochs=DEFAULT_EPOCHS):
    """Trains a single model instance."""
    print(f"Starting training for {model_name}...")
    
    callbacks = get_callbacks(model_name)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def train_all_models(models_dict, train_dataset, val_dataset, epochs=DEFAULT_EPOCHS):
    """Sequentially trains multiple models."""
    history_dict = {}
    
    for name, model in models_dict.items():
        print(f"Training {name}...")
        history = train_single_model(model, name, train_dataset, val_dataset, epochs)
        history_dict[name] = history
        
    return history_dict

def save_training_history(history_dict, output_path='results/training_history.json'):
    """Saves training history to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    serializable_history = {}
    
    for name, history in history_dict.items():
        if hasattr(history, 'history'):
            serializable_history[name] = history.history
        else:
            serializable_history[name] = history
            
    def default_serializer(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'dtype'):
             return float(obj)
        return str(obj)

    try:
        with open(output_path, 'w') as f:
            json.dump(serializable_history, f, default=default_serializer, indent=4)
        print(f"Training history saved to {output_path}")
    except Exception as e:
        print(f"Error saving training history: {e}")

def load_trained_model(model_name, weights_path):
    """Loads a trained model from .h5 weights file."""
    print(f"Loading {model_name} from {weights_path}...")
    try:
        model = load_model(weights_path)
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None
