from data.preprocessing import prepare_dataset

print("Loading Intel Image Classification dataset...")

(X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = prepare_dataset(
    dataset_path='./data_samples', 
    target_size=(150, 150) 
)

if X_train is not None:
    print("\nSuccess! Data integration is correct.")
    print(f"Detected classes ({len(class_names)}): {class_names}")
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
else:
    print("\nFailed. Please check if data_samples folder is empty.")