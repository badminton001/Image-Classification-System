"""
Console user interface for the Image Classification Enhancement System.
"""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    BATCH_SIZE,
    DATASET_PATH,
    EPOCHS,
    MODEL_NAMES,
    MODELS_INFO,
    RESULTS_DIR,
    TARGET_IMAGE_SIZE,
    TEST_SIZE,
    TOP_K,
    TRAIN_SIZE,
    VAL_SIZE,
)
from utils import visualization
from utils.helpers import (
    collect_image_files,
    create_directories,
    format_time,
    print_section,
    print_separator,
    save_results_to_json,
    validate_path,
)
from utils.logger import setup_logger

logger = setup_logger("ImageClassifierUI")


@dataclass
class AppState:
    dataset_loaded: bool = False
    X_train: Any = None
    X_val: Any = None
    X_test: Any = None
    y_train: Any = None
    y_val: Any = None
    y_test: Any = None
    train_data: Any = None
    val_data: Any = None
    test_data: Any = None
    class_names: Optional[List[str]] = None
    models_dict: Dict[str, Any] = field(default_factory=dict)
    histories: Dict[str, Any] = field(default_factory=dict)
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    last_prediction: Any = None


app_state = AppState()


def print_main_menu() -> None:
    print("\n===== Image Classification Enhancement System =====")
    print("1. Load and preprocess dataset")
    print("2. Train a single model")
    print("3. Train all models")
    print("4. Evaluate and compare models")
    print("5. Predict a single image")
    print("6. Batch prediction")
    print("7. Visualizations")
    print("8. Exit")


def get_user_choice() -> int:
    while True:
        try:
            choice = int(input("Enter choice (1-8): ").strip())
            if 1 <= choice <= 8:
                return choice
        except ValueError:
            pass
        print("Invalid input. Please enter a number between 1 and 8.")


def _import_optional(module_name: str, friendly_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"[Error] Missing {friendly_name} module: {module_name}")
        logger.error("%s module not found: %s", friendly_name, module_name)
    except Exception as exc:
        print(f"[Error] Failed to import {friendly_name}: {exc}")
        logger.error("Failed to import %s (%s): %s", friendly_name, module_name, exc)
    return None


def _build_model(model_name: str, input_shape: Any, num_classes: int) -> Optional[Any]:
    arch_module = _import_optional("models.model_architecture", "model architecture")
    if arch_module is None:
        return None

    for fn_name in ("get_model", "build_model", "create_model"):
        if hasattr(arch_module, fn_name):
            builder = getattr(arch_module, fn_name)
            try:
                return builder(model_name=model_name, input_shape=input_shape, num_classes=num_classes)
            except TypeError:
                try:
                    return builder(model_name, input_shape, num_classes)
                except Exception as exc:
                    logger.error("Error when building model using %s: %s", fn_name, exc)
                    print(f"[Error] Failed to build model: {exc}")
                    return None
    print("[Error] Missing model build function (get_model/build_model/create_model).")
    logger.error("Model build function missing in model_architecture.")
    return None


def _train_model(model_name: str, model_obj: Any) -> Optional[Any]:
    trainer = _import_optional("models.train", "training")
    if trainer is None:
        return None

    train_kwargs = dict(
        model=model_obj,
        model_name=model_name,
        X_train=app_state.X_train,
        y_train=app_state.y_train,
        X_val=app_state.X_val,
        y_val=app_state.y_val,
        train_data=app_state.train_data,
        val_data=app_state.val_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    for fn_name in ("train_model", "train", "fit_model"):
        if hasattr(trainer, fn_name):
            fn = getattr(trainer, fn_name)
            try:
                return fn(**train_kwargs)
            except TypeError:
                try:
                    return fn(
                        model_obj,
                        app_state.train_data or app_state.X_train,
                        app_state.y_train,
                        app_state.val_data or app_state.X_val,
                        app_state.y_val,
                    )
                except Exception as exc:
                    logger.error("Training failed with %s: %s", fn_name, exc)
                    print(f"[Error] Training failed: {exc}")
                    return None
    print("[Error] Missing training function (train_model/train/fit_model).")
    logger.error("Train function missing in models.train.")
    return None


def _evaluate_model(model_name: str, model_obj: Any) -> Optional[Dict[str, Any]]:
    evaluator = _import_optional("models.evaluate", "evaluation") or _import_optional("models.evaluation", "evaluation")
    if evaluator is None:
        return None

    eval_kwargs = dict(
        model=model_obj,
        model_name=model_name,
        X_test=app_state.X_test,
        y_test=app_state.y_test,
        test_data=app_state.test_data,
    )

    for fn_name in ("evaluate_model", "evaluate", "run_evaluation"):
        if hasattr(evaluator, fn_name):
            fn = getattr(evaluator, fn_name)
            try:
                return fn(**eval_kwargs)
            except TypeError:
                try:
                    return fn(model_obj, app_state.X_test, app_state.y_test)
                except Exception as exc:
                    logger.error("Evaluation failed with %s: %s", fn_name, exc)
                    print(f"[Error] Evaluation failed: {exc}")
                    return None
    print("[Error] Missing evaluation function (evaluate_model/evaluate/run_evaluation).")
    logger.error("Evaluation function missing in models.evaluate.")
    return None


def _predict_single(model_obj: Any, image_path: Path) -> Optional[Any]:
    predict_module = _import_optional("predict.inference", "prediction") or _import_optional("predict.predictor", "prediction")
    if predict_module is None:
        return None

    candidates = ("predict_single", "predict_image", "run_inference", "predict_one", "predict")
    for fn_name in candidates:
        if hasattr(predict_module, fn_name):
            fn = getattr(predict_module, fn_name)
            try:
                return fn(model_obj, image_path)
            except TypeError:
                try:
                    return fn(image_path)
                except Exception as exc:
                    logger.error("Prediction failed with %s: %s", fn_name, exc)
                    print(f"[Error] Prediction failed: {exc}")
                    return None
    print("[Error] No prediction function found.")
    logger.error("No prediction function found in predict module.")
    return None


def _predict_batch(model_obj: Any, image_paths: List[Path]) -> Optional[Any]:
    predict_module = _import_optional("predict.inference", "prediction") or _import_optional("predict.predictor", "prediction")
    if predict_module is None:
        return None

    candidates = ("predict_batch", "batch_predict", "predict_directory")
    for fn_name in candidates:
        if hasattr(predict_module, fn_name):
            fn = getattr(predict_module, fn_name)
            try:
                return fn(model_obj, image_paths)
            except TypeError:
                try:
                    return fn(image_paths)
                except Exception as exc:
                    logger.error("Batch prediction failed with %s: %s", fn_name, exc)
                    print(f"[Error] Batch prediction failed: {exc}")
                    return None
    print("[Error] No batch prediction function found.")
    logger.error("No batch prediction function found in predict module.")
    return None


def handle_dataset_loading() -> None:
    print_section("Load and preprocess dataset")
    data_module = _import_optional("data.preprocessing", "data preprocessing")
    if data_module is None:
        return

    if not hasattr(data_module, "prepare_dataset"):
        print("[Error] Missing function data.preprocessing.prepare_dataset")
        logger.error("prepare_dataset not found in data.preprocessing")
        return

    try:
        result = data_module.prepare_dataset(
            DATASET_PATH,
            train_size=TRAIN_SIZE,
            val_size=VAL_SIZE,
            test_size=TEST_SIZE,
            target_size=TARGET_IMAGE_SIZE,
        )
    except Exception as exc:
        print(f"[Error] Dataset loading failed: {exc}")
        logger.error("Dataset loading failed: %s", exc)
        return

    if isinstance(result, dict):
        app_state.X_train = result.get("X_train")
        app_state.X_val = result.get("X_val")
        app_state.X_test = result.get("X_test")
        app_state.y_train = result.get("y_train")
        app_state.y_val = result.get("y_val")
        app_state.y_test = result.get("y_test")
        app_state.train_data = result.get("train_data")
        app_state.val_data = result.get("val_data")
        app_state.test_data = result.get("test_data")
        app_state.class_names = result.get("class_names")
    elif isinstance(result, (list, tuple)) and len(result) >= 6:
        app_state.X_train, app_state.X_val, app_state.X_test, app_state.y_train, app_state.y_val, app_state.y_test = result[:6]
        if len(result) >= 7:
            app_state.class_names = result[6]
    else:
        print("[Warning] Unrecognized return format; data may not have loaded correctly.")
        logger.warning("Unrecognized dataset return format: %s", type(result))
        return

    app_state.dataset_loaded = True
    print("Dataset loaded.")
    logger.info("Dataset loaded successfully.")


def _choose_model_name() -> Optional[str]:
    print("Available models:")
    for idx, name in enumerate(MODEL_NAMES, 1):
        info = MODELS_INFO.get(name, {})
        desc = info.get("description", "")
        print(f"{idx}. {name} ({desc})")
    selection = input("Select model number: ").strip()
    if not selection.isdigit():
        print("Invalid input.")
        return None
    index = int(selection)
    if 1 <= index <= len(MODEL_NAMES):
        return MODEL_NAMES[index - 1]
    print("Selection out of range.")
    return None


def _resolve_input_shape() -> Any:
    height, width = TARGET_IMAGE_SIZE
    return (height, width, 3)


def handle_single_model_training() -> None:
    if not app_state.dataset_loaded:
        print("Please load the dataset first (option 1).")
        return
    model_name = _choose_model_name()
    if not model_name:
        return

    num_classes = len(app_state.class_names) if app_state.class_names else 0
    model_obj = _build_model(model_name, _resolve_input_shape(), num_classes)
    if model_obj is None:
        return

    start = time.time()
    history = _train_model(model_name, model_obj)
    duration = time.time() - start
    if history is None:
        print("Training did not complete.")
        return

    app_state.models_dict[model_name] = model_obj
    app_state.histories[model_name] = history
    print(f"{model_name} training finished in {format_time(duration)}.")
    logger.info("Training finished for %s in %s seconds", model_name, round(duration, 2))


def handle_all_models_training() -> None:
    if not app_state.dataset_loaded:
        print("Please load the dataset first (option 1).")
        return
    for model_name in MODEL_NAMES:
        print_separator(f"Training model: {model_name}")
        handle_single_model_training_for(model_name)


def handle_single_model_training_for(model_name: str) -> None:
    num_classes = len(app_state.class_names) if app_state.class_names else 0
    model_obj = _build_model(model_name, _resolve_input_shape(), num_classes)
    if model_obj is None:
        return
    start = time.time()
    history = _train_model(model_name, model_obj)
    duration = time.time() - start
    if history is None:
        return
    app_state.models_dict[model_name] = model_obj
    app_state.histories[model_name] = history
    print(f"{model_name} training finished in {format_time(duration)}.")
    logger.info("Training finished for %s in %s seconds", model_name, round(duration, 2))


def handle_model_evaluation() -> None:
    if not app_state.models_dict:
        print("Please train at least one model first.")
        return

    results: Dict[str, Any] = {}
    for model_name, model_obj in app_state.models_dict.items():
        print_separator(f"Evaluating model: {model_name}")
        metrics = _evaluate_model(model_name, model_obj)
        if metrics is None:
            continue
        results[model_name] = metrics
        print(f"{model_name} evaluation: {metrics}")

    if results:
        app_state.evaluation_results.update(results)
        output_path = save_results_to_json(results, RESULTS_DIR / "evaluation_results.json")
        print(f"Evaluation results saved to {output_path}")
        logger.info("Evaluation results saved to %s", output_path)
    else:
        print("No valid evaluation results produced.")


def _select_trained_model() -> Optional[str]:
    if not app_state.models_dict:
        print("No trained models available.")
        return None
    names = list(app_state.models_dict.keys())
    for idx, name in enumerate(names, 1):
        print(f"{idx}. {name}")
    choice = input("Select model number: ").strip()
    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(names):
            return names[index - 1]
    print("Invalid selection.")
    return None


def handle_single_prediction() -> None:
    if not app_state.models_dict:
        print("Please train or load a model first.")
        return
    model_name = _select_trained_model()
    if not model_name:
        return
    image_input = input("Enter image path: ").strip().strip('"')
    image_path = Path(image_input)
    if not validate_path(image_path):
        return

    prediction = _predict_single(app_state.models_dict[model_name], image_path)
    if prediction is None:
        return

    app_state.last_prediction = prediction
    print(f"Prediction result: {prediction}")
    if isinstance(prediction, dict) and prediction.get("top_k_predictions"):
        top_items = prediction["top_k_predictions"][:TOP_K]
        print("Top-K predictions:")
        for label, score in top_items:
            print(f"- {label}: {score}")
    if isinstance(prediction, dict) and "probabilities" in prediction:
        probs = prediction["probabilities"]
        class_names = prediction.get("class_names", app_state.class_names or [])
        visualization.plot_prediction_distribution(probs, class_names, RESULTS_DIR / "last_prediction.png")
    if input("Generate image with prediction overlay? (y/n): ").strip().lower() == "y":
        try:
            visualization.display_image_with_prediction(
                image_path,
                prediction if isinstance(prediction, dict) else {"predicted_class": prediction},
                RESULTS_DIR / "prediction_overlay.png",
            )
            print("Prediction visualization saved.")
        except Exception as exc:
            logger.error("Failed to visualize prediction: %s", exc)
            print(f"[Error] Visualization failed: {exc}")


def handle_batch_prediction() -> None:
    if not app_state.models_dict:
        print("Please train or load a model first.")
        return
    model_name = _select_trained_model()
    if not model_name:
        return

    dir_input = input("Enter directory containing images: ").strip().strip('"')
    if not validate_path(dir_input):
        return
    image_paths = collect_image_files(dir_input)
    if not image_paths:
        return

    predictions = _predict_batch(app_state.models_dict[model_name], image_paths)
    if predictions is None:
        return

    output_path = save_results_to_json(
        {"files": [str(p) for p in image_paths], "predictions": predictions},
        RESULTS_DIR / "batch_predictions.json",
    )
    print(f"Batch prediction results saved to {output_path}")
    logger.info("Batch predictions saved to %s", output_path)


def _visualization_menu() -> int:
    print("\n===== Visualization Menu =====")
    print("1. Training curves")
    print("2. Confusion matrix")
    print("3. Model comparison")
    print("4. Prediction distribution")
    print("5. Back")
    while True:
        choice = input("Select: ").strip()
        if choice in {"1", "2", "3", "4", "5"}:
            return int(choice)
        print("Invalid input, please choose again.")


def handle_visualization() -> None:
    while True:
        choice = _visualization_menu()
        if choice == 5:
            break
        if choice == 1:
            if not app_state.histories:
                print("No training history available.")
                continue
            visualization.plot_training_history(app_state.histories)
        elif choice == 2:
            if not app_state.evaluation_results:
                print("Please run evaluation first.")
                continue
            for model_name, metrics in app_state.evaluation_results.items():
                cm = metrics.get("confusion_matrix") or metrics.get("cm")
                class_names = metrics.get("class_names") or app_state.class_names or []
                if cm is None:
                    logger.warning("No confusion matrix for %s", model_name)
                    continue
                try:
                    visualization.plot_confusion_matrix(cm, class_names, model_name, RESULTS_DIR / f"{model_name}_cm.png")
                except Exception as exc:
                    logger.error("Failed to plot confusion matrix for %s: %s", model_name, exc)
                    print(f"[Error] Could not plot confusion matrix for {model_name}: {exc}")
        elif choice == 3:
            if not app_state.evaluation_results:
                print("Please run evaluation first.")
                continue
            visualization.plot_model_comparison(app_state.evaluation_results)
        elif choice == 4:
            if not app_state.last_prediction:
                print("No prediction available.")
                continue
            probs = None
            class_names = app_state.class_names or []
            if isinstance(app_state.last_prediction, dict):
                probs = app_state.last_prediction.get("probabilities") or app_state.last_prediction.get("probs")
                class_names = app_state.last_prediction.get("class_names", class_names)
            if probs is None:
                print("Last prediction does not contain probability distribution.")
                continue
            visualization.plot_prediction_distribution(probs, class_names, RESULTS_DIR / "prediction_distribution.png")


def handle_single_model_training_for_cli() -> None:
    """
    Kept for backward compatibility if other modules import this name.
    """
    handle_single_model_training()


def main_menu() -> None:
    create_directories()
    while True:
        print_main_menu()
        choice = get_user_choice()
        if choice == 1:
            handle_dataset_loading()
        elif choice == 2:
            handle_single_model_training()
        elif choice == 3:
            handle_all_models_training()
        elif choice == 4:
            handle_model_evaluation()
        elif choice == 5:
            handle_single_prediction()
        elif choice == 6:
            handle_batch_prediction()
        elif choice == 7:
            handle_visualization()
        elif choice == 8:
            print("Goodbye!")
            break


__all__ = [
    "print_main_menu",
    "get_user_choice",
    "handle_dataset_loading",
    "handle_single_model_training",
    "handle_all_models_training",
    "handle_model_evaluation",
    "handle_single_prediction",
    "handle_batch_prediction",
    "handle_visualization",
    "main_menu",
]

