
"""
Tkinter GUI for the Image Classification Enhancement System.
"""
from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from config import (
    BATCH_SIZE,
    DATASET_PATH,
    EPOCHS,
    LEARNING_RATE,
    MODEL_NAMES,
    RESULTS_DIR,
    TARGET_IMAGE_SIZE,
    TEST_SIZE,
    TOP_K,
    TRAIN_SIZE,
    VAL_SIZE,
)
from utils import visualization
from utils.helpers import collect_image_files, create_directories, format_time, save_results_to_json
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
    last_prediction_image: Optional[Path] = None


class ImageClassifierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.state = AppState()
        self._busy = False
        self._action_buttons: List[ttk.Button] = []

        self.dataset_path_var = tk.StringVar(value=str(DATASET_PATH))
        self.train_model_var = tk.StringVar(value=MODEL_NAMES[0] if MODEL_NAMES else "")
        self.predict_model_var = tk.StringVar(value="")
        self.image_path_var = tk.StringVar(value="")
        self.batch_dir_var = tk.StringVar(value="")
        self.top_k_var = tk.StringVar(value=str(TOP_K))
        self.save_overlay_var = tk.BooleanVar(value=True)
        self.save_distribution_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready.")

        self.log_text: Optional[scrolledtext.ScrolledText] = None
        self.progress: Optional[ttk.Progressbar] = None

        self._build_ui()
        self._refresh_model_options()

    def _build_ui(self) -> None:
        self.root.title("Image Classification Enhancement System")
        self.root.geometry("1100x720")
        self.root.minsize(1000, 680)

        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left_frame.columnconfigure(0, weight=1)

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        title_label = ttk.Label(
            left_frame,
            text="Image Classification Enhancement System",
            font=("Segoe UI", 14, "bold"),
        )
        title_label.grid(row=0, column=0, sticky="w", pady=(0, 10))

        dataset_frame = ttk.LabelFrame(left_frame, text="Dataset")
        dataset_frame.grid(row=1, column=0, sticky="ew", pady=6)
        dataset_frame.columnconfigure(1, weight=1)
        ttk.Label(dataset_frame, text="Dataset path:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(dataset_frame, textvariable=self.dataset_path_var).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(dataset_frame, text="Browse", command=self._browse_dataset).grid(row=0, column=2, padx=6, pady=4)
        load_btn = ttk.Button(dataset_frame, text="Load Dataset", command=self._on_load_dataset)
        load_btn.grid(row=1, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
        self._action_buttons.append(load_btn)

        training_frame = ttk.LabelFrame(left_frame, text="Training")
        training_frame.grid(row=2, column=0, sticky="ew", pady=6)
        training_frame.columnconfigure(1, weight=1)
        ttk.Label(training_frame, text="Model:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.train_model_combo = ttk.Combobox(
            training_frame,
            textvariable=self.train_model_var,
            values=MODEL_NAMES,
            state="readonly",
        )
        self.train_model_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        train_btn = ttk.Button(training_frame, text="Train Selected Model", command=self._on_train_single)
        train_btn.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        train_all_btn = ttk.Button(training_frame, text="Train All Models", command=self._on_train_all)
        train_all_btn.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        self._action_buttons.extend([train_btn, train_all_btn])

        eval_frame = ttk.LabelFrame(left_frame, text="Evaluation")
        eval_frame.grid(row=3, column=0, sticky="ew", pady=6)
        eval_btn = ttk.Button(eval_frame, text="Evaluate Models", command=self._on_evaluate_models)
        eval_btn.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        self._action_buttons.append(eval_btn)

        predict_frame = ttk.LabelFrame(left_frame, text="Prediction")
        predict_frame.grid(row=4, column=0, sticky="ew", pady=6)
        predict_frame.columnconfigure(1, weight=1)
        ttk.Label(predict_frame, text="Model:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.predict_model_combo = ttk.Combobox(
            predict_frame,
            textvariable=self.predict_model_var,
            values=MODEL_NAMES,
            state="readonly",
        )
        self.predict_model_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Label(predict_frame, text="Top-K:").grid(row=0, column=2, sticky="w", padx=6, pady=4)
        ttk.Entry(predict_frame, textvariable=self.top_k_var, width=6).grid(row=0, column=3, sticky="w", padx=6, pady=4)

        ttk.Label(predict_frame, text="Image path:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(predict_frame, textvariable=self.image_path_var).grid(row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=4)
        ttk.Button(predict_frame, text="Browse", command=self._browse_image).grid(row=1, column=3, padx=6, pady=4)
        predict_btn = ttk.Button(predict_frame, text="Predict Image", command=self._on_predict_single)
        predict_btn.grid(row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=4)

        ttk.Label(predict_frame, text="Batch folder:").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(predict_frame, textvariable=self.batch_dir_var).grid(row=3, column=1, columnspan=2, sticky="ew", padx=6, pady=4)
        ttk.Button(predict_frame, text="Browse", command=self._browse_batch_dir).grid(row=3, column=3, padx=6, pady=4)
        batch_btn = ttk.Button(predict_frame, text="Batch Predict", command=self._on_batch_predict)
        batch_btn.grid(row=4, column=0, columnspan=4, sticky="ew", padx=6, pady=4)

        ttk.Checkbutton(predict_frame, text="Save overlay image", variable=self.save_overlay_var).grid(row=5, column=0, columnspan=2, sticky="w", padx=6, pady=2)
        ttk.Checkbutton(predict_frame, text="Save prediction distribution", variable=self.save_distribution_var).grid(row=5, column=2, columnspan=2, sticky="w", padx=6, pady=2)
        self._action_buttons.extend([predict_btn, batch_btn])

        viz_frame = ttk.LabelFrame(left_frame, text="Visualization")
        viz_frame.grid(row=5, column=0, sticky="ew", pady=6)
        viz_frame.columnconfigure(0, weight=1)
        viz_buttons = [
            ("Training History", self._on_plot_training_history),
            ("Confusion Matrix", self._on_plot_confusion_matrix),
            ("Model Comparison", self._on_plot_model_comparison),
            ("Prediction Distribution", self._on_plot_prediction_distribution),
            ("Prediction Overlay", self._on_plot_prediction_overlay),
        ]
        for idx, (label, handler) in enumerate(viz_buttons):
            btn = ttk.Button(viz_frame, text=label, command=handler)
            btn.grid(row=idx, column=0, sticky="ew", padx=6, pady=3)
            self._action_buttons.append(btn)

        log_frame = ttk.LabelFrame(right_frame, text="Activity Log")
        log_frame.grid(row=0, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap="word", height=30, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        clear_btn = ttk.Button(log_frame, text="Clear Log", command=self._clear_log)
        clear_btn.grid(row=1, column=0, sticky="e", padx=6, pady=(0, 6))

        status_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        status_frame.grid(row=1, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate")
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=1, sticky="e")

    def _browse_dataset(self) -> None:
        path = filedialog.askdirectory(title="Select Dataset Folder")
        if path:
            self.dataset_path_var.set(path)

    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")],
        )
        if path:
            self.image_path_var.set(path)

    def _browse_batch_dir(self) -> None:
        path = filedialog.askdirectory(title="Select Batch Folder")
        if path:
            self.batch_dir_var.set(path)

    def _clear_log(self) -> None:
        if self.log_text is None:
            return
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _log(self, message: str) -> None:
        if self.log_text is None:
            return
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _set_busy(self, busy: bool, message: Optional[str] = None) -> None:
        self._busy = busy
        for btn in self._action_buttons:
            btn.configure(state="disabled" if busy else "normal")
        if self.progress:
            if busy:
                self.progress.start(10)
            else:
                self.progress.stop()
        if message:
            self._set_status(message)
        elif not busy:
            self._set_status("Ready.")

    def _post(self, func, *args, **kwargs) -> None:
        self.root.after(0, lambda: func(*args, **kwargs))

    def _run_task(self, label: str, func) -> None:
        if self._busy:
            messagebox.showinfo("Busy", "An operation is already running.")
            return

        def worker() -> None:
            self._post(self._set_busy, True, label)
            try:
                func()
            except Exception as exc:
                logger.error("Operation failed: %s", exc)
                self._post(messagebox.showerror, "Error", str(exc))
            finally:
                self._post(self._set_busy, False, None)

        threading.Thread(target=worker, daemon=True).start()

    def _import_optional(self, module_name: str, friendly_name: str) -> Optional[Any]:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            self._post(self._log, f"[Error] Missing {friendly_name} module: {module_name}")
            logger.error("%s module not found: %s", friendly_name, module_name)
        except Exception as exc:
            self._post(self._log, f"[Error] Failed to import {friendly_name}: {exc}")
            logger.error("Failed to import %s (%s): %s", friendly_name, module_name, exc)
        return None

    def _resolve_num_classes(self) -> int:
        if self.state.class_names:
            return len(self.state.class_names)
        try:
            config_models = importlib.import_module("models.config_models")
            return int(getattr(config_models, "NUM_CLASSES", 0))
        except Exception:
            return 0

    def _resolve_class_names(self) -> List[str]:
        if self.state.class_names:
            return list(self.state.class_names)
        try:
            config_models = importlib.import_module("models.config_models")
            names = getattr(config_models, "CLASS_NAMES", None)
            return list(names) if names else []
        except Exception:
            return []

    def _update_state_from_dataset(self, result: Any) -> bool:
        if result is None:
            return False

        if isinstance(result, dict):
            self.state.X_train = result.get("X_train")
            self.state.X_val = result.get("X_val")
            self.state.X_test = result.get("X_test")
            self.state.y_train = result.get("y_train")
            self.state.y_val = result.get("y_val")
            self.state.y_test = result.get("y_test")
            self.state.train_data = result.get("train_data")
            self.state.val_data = result.get("val_data")
            self.state.test_data = result.get("test_data")
            self.state.class_names = result.get("class_names")
            return True

        if isinstance(result, (list, tuple)):
            if len(result) == 4 and all(isinstance(part, (list, tuple)) for part in result[:3]):
                (x_train, y_train), (x_val, y_val), (x_test, y_test), class_names = result
                self.state.X_train = x_train
                self.state.X_val = x_val
                self.state.X_test = x_test
                self.state.y_train = y_train
                self.state.y_val = y_val
                self.state.y_test = y_test
                self.state.class_names = list(class_names) if class_names is not None else None
                return True
            if len(result) >= 6:
                self.state.X_train, self.state.X_val, self.state.X_test, self.state.y_train, self.state.y_val, self.state.y_test = result[:6]
                if len(result) >= 7:
                    self.state.class_names = list(result[6]) if result[6] is not None else None
                return True

        return False

    def _resolve_input_shape(self) -> Tuple[int, int, int]:
        height, width = TARGET_IMAGE_SIZE
        return (height, width, 3)

    def _prepare_training_data(self) -> Tuple[Any, Any, int]:
        if self.state.X_train is None or self.state.y_train is None:
            raise ValueError("Training data not loaded.")
        if self.state.X_val is None or self.state.y_val is None:
            raise ValueError("Validation data not loaded.")

        num_classes = self._resolve_num_classes()
        if num_classes <= 0:
            raise ValueError("Number of classes is unknown.")

        y_train = self.state.y_train
        y_val = self.state.y_val
        y_train_cat = None
        y_val_cat = None

        try:
            from tensorflow.keras.utils import to_categorical

            y_train_cat = to_categorical(y_train, num_classes)
            y_val_cat = to_categorical(y_val, num_classes)
        except Exception as exc:
            logger.warning("Failed to convert labels to categorical: %s", exc)

        augmentation = self._import_optional("data.augmentation", "data augmentation")
        if augmentation and y_train_cat is not None and y_val_cat is not None:
            if hasattr(augmentation, "create_augmented_train_generator") and hasattr(augmentation, "create_validation_generator"):
                train_dataset = augmentation.create_augmented_train_generator(self.state.X_train, y_train_cat, BATCH_SIZE)
                val_dataset = augmentation.create_validation_generator(self.state.X_val, y_val_cat, BATCH_SIZE)
                return train_dataset, val_dataset, num_classes

        train_dataset = (self.state.X_train, y_train_cat if y_train_cat is not None else y_train)
        val_dataset = (self.state.X_val, y_val_cat if y_val_cat is not None else y_val)
        return train_dataset, val_dataset, num_classes

    def _build_model(self, model_name: str, num_classes: int) -> Optional[Any]:
        arch_module = self._import_optional("models.model_architecture", "model architecture")
        if arch_module is None:
            return None

        builder = None
        for fn_name in ("get_model", "build_model", "create_model"):
            if hasattr(arch_module, fn_name):
                builder = getattr(arch_module, fn_name)
                break

        if builder is None:
            lower_name = model_name.lower()
            mapping = {
                "vgg16": "build_vgg16_model",
                "resnet50": "build_resnet50_model",
                "mobilenetv2": "build_mobilenetv2_model",
            }
            fn_name = mapping.get(lower_name)
            if fn_name and hasattr(arch_module, fn_name):
                builder = getattr(arch_module, fn_name)

        if builder is None:
            self._post(self._log, "[Error] Missing model build function.")
            logger.error("Model build function missing in model_architecture.")
            return None

        try:
            model_obj = builder(model_name=model_name, input_shape=self._resolve_input_shape(), num_classes=num_classes)
        except TypeError:
            try:
                model_obj = builder(num_classes, self._resolve_input_shape())
            except TypeError:
                model_obj = builder(num_classes)
        except Exception as exc:
            logger.error("Model build failed: %s", exc)
            self._post(self._log, f"[Error] Model build failed: {exc}")
            return None

        if hasattr(arch_module, "compile_model"):
            try:
                model_obj = arch_module.compile_model(model_obj, learning_rate=LEARNING_RATE)
            except TypeError:
                model_obj = arch_module.compile_model(model_obj)
            except Exception as exc:
                logger.error("Model compile failed: %s", exc)
                self._post(self._log, f"[Error] Model compile failed: {exc}")
                return None
        return model_obj

    def _train_model(self, model_name: str, model_obj: Any) -> Optional[Any]:
        trainer = self._import_optional("models.train", "training")
        if trainer is None:
            return None

        train_dataset, val_dataset, _ = self._prepare_training_data()

        for fn_name in ("train_single_model", "train_model", "train", "fit_model"):
            if hasattr(trainer, fn_name):
                fn = getattr(trainer, fn_name)
                try:
                    return fn(model_obj, model_name, train_dataset, val_dataset, epochs=EPOCHS)
                except TypeError:
                    try:
                        return fn(model_obj, train_dataset, val_dataset, EPOCHS)
                    except TypeError:
                        try:
                            return fn(
                                model_obj,
                                self.state.train_data or train_dataset,
                                self.state.y_train,
                                self.state.val_data or val_dataset,
                                self.state.y_val,
                            )
                        except Exception as exc:
                            logger.error("Training failed with %s: %s", fn_name, exc)
                            self._post(self._log, f"[Error] Training failed: {exc}")
                            return None
        self._post(self._log, "[Error] Missing training function.")
        logger.error("Train function missing in models.train.")
        return None

    def _evaluate_models(self) -> Optional[Dict[str, Any]]:
        evaluator = self._import_optional("models.evaluate", "evaluation")
        if evaluator is None:
            return None

        if hasattr(evaluator, "evaluate_all_models"):
            try:
                class_names = self._resolve_class_names()
                return evaluator.evaluate_all_models(self.state.models_dict, self.state.X_test, self.state.y_test, class_names)
            except Exception as exc:
                logger.error("Evaluation failed: %s", exc)
                self._post(self._log, f"[Error] Evaluation failed: {exc}")
                return None

        results: Dict[str, Any] = {}
        for model_name, model_obj in self.state.models_dict.items():
            for fn_name in ("evaluate_model", "evaluate", "run_evaluation", "evaluate_single_model"):
                if hasattr(evaluator, fn_name):
                    fn = getattr(evaluator, fn_name)
                    try:
                        results[model_name] = fn(model_obj, self.state.X_test, self.state.y_test)
                    except Exception as exc:
                        logger.error("Evaluation failed with %s: %s", fn_name, exc)
                        self._post(self._log, f"[Error] Evaluation failed: {exc}")
                    break
        return results or None

    def _predict_single(self, model_obj: Any, image_path: Path, top_k: int) -> Optional[Any]:
        predict_module = self._import_optional("predict.inference", "prediction")
        if predict_module is None:
            return None

        class_names = self._resolve_class_names()
        candidates = ("predict_single_image", "predict_single", "predict_image", "run_inference", "predict_one", "predict")
        for fn_name in candidates:
            if hasattr(predict_module, fn_name):
                fn = getattr(predict_module, fn_name)
                try:
                    result = fn(model=model_obj, image_path=str(image_path), class_names=class_names, top_k=top_k)
                except TypeError:
                    try:
                        result = fn(model_obj, str(image_path), class_names, top_k)
                    except Exception as exc:
                        logger.error("Prediction failed with %s: %s", fn_name, exc)
                        self._post(self._log, f"[Error] Prediction failed: {exc}")
                        return None

                if isinstance(result, dict) and "probabilities" not in result:
                    if hasattr(predict_module, "preprocess_input_image"):
                        try:
                            img = predict_module.preprocess_input_image(str(image_path), target_size=TARGET_IMAGE_SIZE)
                            probs = model_obj.predict(img, verbose=0)[0]
                            result["probabilities"] = probs
                            result["class_names"] = class_names
                        except Exception as exc:
                            logger.warning("Failed to compute probabilities: %s", exc)
                return result
        self._post(self._log, "[Error] No prediction function found.")
        logger.error("No prediction function found in predict module.")
        return None

    def _predict_batch(self, model_obj: Any, image_paths: Sequence[Path]) -> Optional[Any]:
        predict_module = self._import_optional("predict.inference", "prediction")
        if predict_module is None:
            return None

        class_names = self._resolve_class_names()
        candidates = ("predict_batch", "batch_predict", "predict_directory")
        for fn_name in candidates:
            if hasattr(predict_module, fn_name):
                fn = getattr(predict_module, fn_name)
                try:
                    return fn(model=model_obj, image_paths=[str(p) for p in image_paths], class_names=class_names)
                except TypeError:
                    try:
                        return fn(model_obj, [str(p) for p in image_paths], class_names)
                    except Exception as exc:
                        logger.error("Batch prediction failed with %s: %s", fn_name, exc)
                        self._post(self._log, f"[Error] Batch prediction failed: {exc}")
                        return None
        self._post(self._log, "[Error] No batch prediction function found.")
        logger.error("No batch prediction function found in predict module.")
        return None

    def _refresh_model_options(self) -> None:
        trained_models = list(self.state.models_dict.keys())
        if trained_models:
            self.predict_model_combo.configure(values=trained_models)
            if self.predict_model_var.get() not in trained_models:
                self.predict_model_var.set(trained_models[0])
        else:
            self.predict_model_combo.configure(values=MODEL_NAMES)
            if MODEL_NAMES and not self.predict_model_var.get():
                self.predict_model_var.set(MODEL_NAMES[0])

    def _on_load_dataset(self) -> None:
        def task() -> None:
            dataset_path = Path(self.dataset_path_var.get().strip())
            if not dataset_path.exists():
                self._post(messagebox.showwarning, "Missing Path", f"Dataset path does not exist: {dataset_path}")
                return

            data_module = self._import_optional("data.preprocessing", "data preprocessing")
            if data_module is None:
                return
            if not hasattr(data_module, "prepare_dataset"):
                self._post(messagebox.showerror, "Missing Function", "prepare_dataset not found in data.preprocessing")
                return

            self._post(self._log, f"Loading dataset from {dataset_path}...")
            try:
                try:
                    result = data_module.prepare_dataset(
                        dataset_path,
                        train_size=TRAIN_SIZE,
                        val_size=VAL_SIZE,
                        test_size=TEST_SIZE,
                        target_size=TARGET_IMAGE_SIZE,
                    )
                except TypeError:
                    result = data_module.prepare_dataset(
                        dataset_path,
                        target_size=TARGET_IMAGE_SIZE,
                        test_size=TEST_SIZE,
                        val_size=VAL_SIZE,
                    )
            except Exception as exc:
                logger.error("Dataset loading failed: %s", exc)
                self._post(messagebox.showerror, "Error", f"Dataset loading failed: {exc}")
                return

            if not self._update_state_from_dataset(result):
                self._post(messagebox.showwarning, "Warning", "Unrecognized dataset format returned.")
                return

            self.state.dataset_loaded = True
            self._post(self._log, "Dataset loaded successfully.")
            self._post(messagebox.showinfo, "Success", "Dataset loaded successfully.")

        self._run_task("Loading dataset...", task)

    def _on_train_single(self) -> None:
        def task() -> None:
            if not self.state.dataset_loaded:
                self._post(messagebox.showwarning, "Missing Data", "Please load the dataset first.")
                return

            model_name = self.train_model_var.get().strip()
            if not model_name:
                self._post(messagebox.showwarning, "Missing Model", "Please select a model.")
                return

            num_classes = self._resolve_num_classes()
            if num_classes <= 0:
                self._post(messagebox.showwarning, "Missing Classes", "Unable to determine number of classes.")
                return

            self._post(self._log, f"Building model {model_name}...")
            model_obj = self._build_model(model_name, num_classes)
            if model_obj is None:
                return

            self._post(self._log, f"Training {model_name}...")
            start = time.time()
            history = self._train_model(model_name, model_obj)
            if history is None:
                return
            duration = time.time() - start

            self.state.models_dict[model_name] = model_obj
            self.state.histories[model_name] = history
            self._refresh_model_options()
            self._post(self._log, f"{model_name} training finished in {format_time(duration)}.")
            self._post(messagebox.showinfo, "Training Complete", f"{model_name} training finished.")

        self._run_task("Training model...", task)

    def _on_train_all(self) -> None:
        def task() -> None:
            if not self.state.dataset_loaded:
                self._post(messagebox.showwarning, "Missing Data", "Please load the dataset first.")
                return

            num_classes = self._resolve_num_classes()
            if num_classes <= 0:
                self._post(messagebox.showwarning, "Missing Classes", "Unable to determine number of classes.")
                return

            for model_name in MODEL_NAMES:
                self._post(self._log, f"Training {model_name}...")
                model_obj = self._build_model(model_name, num_classes)
                if model_obj is None:
                    continue
                start = time.time()
                history = self._train_model(model_name, model_obj)
                duration = time.time() - start
                if history is None:
                    continue
                self.state.models_dict[model_name] = model_obj
                self.state.histories[model_name] = history
                self._post(self._log, f"{model_name} training finished in {format_time(duration)}.")

            self._refresh_model_options()
            self._post(messagebox.showinfo, "Training Complete", "Finished training all available models.")

        self._run_task("Training all models...", task)

    def _on_evaluate_models(self) -> None:
        def task() -> None:
            if not self.state.models_dict:
                self._post(messagebox.showwarning, "Missing Models", "Please train at least one model first.")
                return

            if self.state.X_test is None or self.state.y_test is None:
                self._post(messagebox.showwarning, "Missing Data", "Test data is not available.")
                return

            results = self._evaluate_models()
            if results:
                self.state.evaluation_results.update(results)
                output_path = save_results_to_json(results, RESULTS_DIR / "evaluation_results.json")
                self._post(self._log, f"Evaluation results saved to {output_path}")
                self._post(messagebox.showinfo, "Evaluation Complete", "Evaluation results saved.")
            else:
                self._post(messagebox.showwarning, "Evaluation", "No valid evaluation results produced.")

        self._run_task("Evaluating models...", task)

    def _on_predict_single(self) -> None:
        def task() -> None:
            model_name = self.predict_model_var.get().strip()
            if model_name not in self.state.models_dict:
                self._post(messagebox.showwarning, "Missing Model", "Please train the selected model first.")
                return

            image_path = Path(self.image_path_var.get().strip())
            if not image_path.exists():
                self._post(messagebox.showwarning, "Missing Image", "Please select a valid image path.")
                return

            try:
                top_k = max(1, int(self.top_k_var.get().strip()))
            except ValueError:
                top_k = TOP_K

            self._post(self._log, f"Predicting {image_path.name} with {model_name}...")
            prediction = self._predict_single(self.state.models_dict[model_name], image_path, top_k)
            if prediction is None:
                return

            self.state.last_prediction = prediction
            self.state.last_prediction_image = image_path
            self._post(self._log, f"Prediction result: {prediction}")

            if self.save_distribution_var.get() and isinstance(prediction, dict):
                probs = prediction.get("probabilities") or prediction.get("probs")
                if probs is not None:
                    visualization.plot_prediction_distribution(probs, self._resolve_class_names(), RESULTS_DIR / "last_prediction.png")
                    self._post(self._log, "Prediction distribution saved.")

            if self.save_overlay_var.get():
                try:
                    visualization.display_image_with_prediction(
                        image_path,
                        prediction if isinstance(prediction, dict) else {"predicted_class": prediction},
                        RESULTS_DIR / "prediction_overlay.png",
                    )
                    self._post(self._log, "Prediction overlay saved.")
                except Exception as exc:
                    logger.error("Failed to render prediction overlay: %s", exc)
                    self._post(self._log, f"[Error] Prediction overlay failed: {exc}")

            self._post(messagebox.showinfo, "Prediction Complete", "Prediction finished.")

        self._run_task("Predicting image...", task)

    def _on_batch_predict(self) -> None:
        def task() -> None:
            model_name = self.predict_model_var.get().strip()
            if model_name not in self.state.models_dict:
                self._post(messagebox.showwarning, "Missing Model", "Please train the selected model first.")
                return

            batch_dir = Path(self.batch_dir_var.get().strip())
            if not batch_dir.exists():
                self._post(messagebox.showwarning, "Missing Folder", "Please select a valid batch folder.")
                return

            image_paths = collect_image_files(batch_dir)
            if not image_paths:
                self._post(messagebox.showwarning, "No Images", "No supported images found in the selected folder.")
                return

            self._post(self._log, f"Batch predicting {len(image_paths)} images...")
            predictions = self._predict_batch(self.state.models_dict[model_name], image_paths)
            if predictions is None:
                return

            output_path = save_results_to_json(
                {"files": [str(p) for p in image_paths], "predictions": predictions},
                RESULTS_DIR / "batch_predictions.json",
            )
            self._post(self._log, f"Batch predictions saved to {output_path}")
            self._post(messagebox.showinfo, "Batch Prediction", "Batch predictions saved.")

        self._run_task("Batch prediction...", task)

    def _on_plot_training_history(self) -> None:
        if not self.state.histories:
            messagebox.showwarning("Missing Data", "No training history available.")
            return
        visualization.plot_training_history(self.state.histories)
        self._log("Training history plot saved.")

    def _on_plot_confusion_matrix(self) -> None:
        if not self.state.evaluation_results:
            messagebox.showwarning("Missing Data", "Please evaluate models first.")
            return
        for model_name, metrics in self.state.evaluation_results.items():
            cm = metrics.get("confusion_matrix") or metrics.get("cm")
            class_names = metrics.get("class_names") or self._resolve_class_names()
            if cm is None:
                self._log(f"[Warning] No confusion matrix for {model_name}.")
                continue
            try:
                visualization.plot_confusion_matrix(cm, class_names, model_name, RESULTS_DIR / f"{model_name}_cm.png")
                self._log(f"Confusion matrix saved for {model_name}.")
            except Exception as exc:
                logger.error("Failed to plot confusion matrix for %s: %s", model_name, exc)
                self._log(f"[Error] Confusion matrix failed for {model_name}: {exc}")

    def _on_plot_model_comparison(self) -> None:
        if not self.state.evaluation_results:
            messagebox.showwarning("Missing Data", "Please evaluate models first.")
            return
        visualization.plot_model_comparison(self.state.evaluation_results)
        self._log("Model comparison plot saved.")

    def _on_plot_prediction_distribution(self) -> None:
        prediction = self.state.last_prediction
        if not prediction:
            messagebox.showwarning("Missing Data", "No prediction results available.")
            return
        if isinstance(prediction, dict):
            probs = prediction.get("probabilities") or prediction.get("probs")
        else:
            probs = None
        if probs is None:
            messagebox.showwarning("Missing Data", "Prediction probabilities not available.")
            return
        visualization.plot_prediction_distribution(probs, self._resolve_class_names(), RESULTS_DIR / "prediction_distribution.png")
        self._log("Prediction distribution plot saved.")

    def _on_plot_prediction_overlay(self) -> None:
        if not self.state.last_prediction or not self.state.last_prediction_image:
            messagebox.showwarning("Missing Data", "No prediction results available.")
            return
        try:
            visualization.display_image_with_prediction(
                self.state.last_prediction_image,
                self.state.last_prediction if isinstance(self.state.last_prediction, dict) else {"predicted_class": self.state.last_prediction},
                RESULTS_DIR / "prediction_overlay.png",
            )
            self._log("Prediction overlay saved.")
        except Exception as exc:
            logger.error("Failed to render prediction overlay: %s", exc)
            messagebox.showerror("Error", f"Prediction overlay failed: {exc}")


def launch_gui() -> None:
    create_directories()
    root = tk.Tk()
    app = ImageClassifierApp(root)
    app._log("Application started.")
    root.mainloop()


def main_menu() -> None:
    """
    Backward-compatible entry point that launches the Tkinter GUI.
    """
    launch_gui()


__all__ = ["ImageClassifierApp", "launch_gui", "main_menu"]
