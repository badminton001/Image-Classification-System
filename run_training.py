import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from data.preprocessing import prepare_dataset
from data.augmentation import create_augmented_train_generator, create_validation_generator
from models.config_models import NUM_CLASSES, AVAILABLE_MODELS, DEFAULT_EPOCHS
from models import (
    build_vgg16_model, 
    build_resnet50_model, 
    build_mobilenetv2_model,
    compile_model,
    train_all_models,
    save_training_history
)

# Configuration
DATASET_PATH = './data_samples' 
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)

def main():
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' is invalid.")
        return

    # 1. Data Processing
    print("Loading and Preprocessing Data...")
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = prepare_dataset(
            DATASET_PATH, 
            target_size=TARGET_SIZE
        )
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    if X_train is None:
        print("Failed to load data.")
        return

    # Label Encoding
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)

    # 2. Generators
    print("Creating Data Generators...")
    train_gen = create_augmented_train_generator(X_train, y_train_cat, BATCH_SIZE)
    val_gen = create_validation_generator(X_val, y_val_cat, BATCH_SIZE)

    # 3. Model Initialization
    print("Initializing Models...")
    models = {}
    
    # VGG16
    vgg = build_vgg16_model(NUM_CLASSES)
    models['vgg16'] = compile_model(vgg)
    
    # ResNet50
    resnet = build_resnet50_model(NUM_CLASSES)
    models['resnet50'] = compile_model(resnet)
    
    # MobileNetV2
    mobilenet = build_mobilenetv2_model(NUM_CLASSES)
    models['mobilenetv2'] = compile_model(mobilenet)

    # 4. Training
    print("Starting Training...")
    history_dict = train_all_models(models, train_gen, val_gen, epochs=DEFAULT_EPOCHS)
    
    # 5. Save Results
    save_training_history(history_dict)
    print("Training finished.")

if __name__ == "__main__":
    main()
