import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from .data_loader import load_dataset_from_directory, load_image

def resize_image(image, target_size=(224, 224)):
    return image.resize(target_size, Image.LANCZOS)

def img_to_array(image):
    return np.array(image)

def normalize_image(image_array):
    return image_array.astype('float32') / 255.0

def split_dataset(image_paths, labels, test_size=0.15, val_size=0.15):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    relative_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, stratify=y_train_val, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def _load_and_process_batch(image_paths, target_size):
    data_list = []
    for path in image_paths:
        img = load_image(path)
        if img:
            img = resize_image(img, target_size)
            img_arr = img_to_array(img)
            img_arr = normalize_image(img_arr)
            data_list.append(img_arr)
            
    return np.array(data_list)

def prepare_dataset(dataset_path, target_size=(224, 224), test_size=0.15, val_size=0.15):
    print(f"[Info] Scanning dataset from: {dataset_path}")
    image_paths, labels, class_names = load_dataset_from_directory(dataset_path)
    
    if len(image_paths) == 0:
        print("[Error] No images found!")
        return None, None, None, None

    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = split_dataset(
        image_paths, labels, test_size, val_size
    )
    
    print(f"[Info] Split results: Train={len(X_train_paths)}, Val={len(X_val_paths)}, Test={len(X_test_paths)}")
    
    print("[Info] Processing Training set...")
    X_train = _load_and_process_batch(X_train_paths, target_size)
    
    print("[Info] Processing Validation set...")
    X_val = _load_and_process_batch(X_val_paths, target_size)
    
    print("[Info] Processing Test set...")
    X_test = _load_and_process_batch(X_test_paths, target_size)
    
    print("[Info] All data prepared successfully.")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names