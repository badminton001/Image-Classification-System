
"""
Tkinter GUI for the Image Classification Enhancement System.
"""
from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from config import DATASET_PATH, RESULTS_DIR, TARGET_IMAGE_SIZE, TOP_K, WEIGHTS_DIR
from utils import visualization
from utils.helpers import collect_image_files, create_directories, save_results_to_json
from utils.logger import setup_logger

logger = setup_logger("ImageClassifierUI")

WEIGHT_EXTENSIONS = (".h5", ".keras", ".hdf5")


@dataclass
class AppState:
    model: Any = None
    model_name: str = ""
    model_path: Optional[Path] = None
    class_names: List[str] = field(default_factory=list)
    last_prediction: Any = None
    last_prediction_image: Optional[Path] = None


class ImageClassifierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.state = AppState()
        self._busy = False
        self._action_buttons: List[ttk.Button] = []
        self.weights_files: List[Path] = []

        self.dataset_path_var = tk.StringVar(value=str(DATASET_PATH))
        self.weights_var = tk.StringVar(value="")
        self.class_names_var = tk.StringVar(value="")
        self.image_path_var = tk.StringVar(value="")
        self.batch_dir_var = tk.StringVar(value="")
        self.top_k_var = tk.StringVar(value=str(TOP_K))
        self.save_overlay_var = tk.BooleanVar(value=True)
        self.save_distribution_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready.")
        self.model_status_var = tk.StringVar(value="No model loaded.")

        self.log_text: Optional[scrolledtext.ScrolledText] = None
        self.progress: Optional[ttk.Progressbar] = None
        self.weights_combo: Optional[ttk.Combobox] = None
        self.load_model_btn: Optional[ttk.Button] = None

        self._build_ui()
        self._refresh_weights()

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

        prediction_frame = ttk.LabelFrame(left_frame, text="Prediction")
        prediction_frame.grid(row=1, column=0, sticky="ew", pady=6)
        prediction_frame.columnconfigure(1, weight=1)
        ttk.Label(prediction_frame, text="Weights file:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.weights_combo = ttk.Combobox(prediction_frame, textvariable=self.weights_var, values=[], state="readonly")
        self.weights_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(prediction_frame, text="Browse", command=self._browse_weights).grid(row=0, column=2, padx=6, pady=4)
        ttk.Button(prediction_frame, text="Refresh", command=self._refresh_weights).grid(row=0, column=3, padx=6, pady=4)

        self.load_model_btn = ttk.Button(prediction_frame, text="Load Model", command=self._on_load_model)
        self.load_model_btn.grid(row=1, column=0, columnspan=4, sticky="ew", padx=6, pady=4)
        ttk.Label(prediction_frame, textvariable=self.model_status_var).grid(row=2, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 6))

        ttk.Label(prediction_frame, text="Top-K:").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(prediction_frame, textvariable=self.top_k_var, width=6).grid(row=3, column=1, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(prediction_frame, text="Save overlay image", variable=self.save_overlay_var).grid(row=3, column=2, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(prediction_frame, text="Save prediction distribution", variable=self.save_distribution_var).grid(row=3, column=3, sticky="w", padx=6, pady=4)

        ttk.Label(prediction_frame, text="Image path:").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(prediction_frame, textvariable=self.image_path_var).grid(row=4, column=1, columnspan=2, sticky="ew", padx=6, pady=4)
        ttk.Button(prediction_frame, text="Browse", command=self._browse_image).grid(row=4, column=3, padx=6, pady=4)
        predict_btn = ttk.Button(prediction_frame, text="Predict Image", command=self._on_predict_single)
        predict_btn.grid(row=5, column=0, columnspan=4, sticky="ew", padx=6, pady=4)

        ttk.Label(prediction_frame, text="Batch folder:").grid(row=6, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(prediction_frame, textvariable=self.batch_dir_var).grid(row=6, column=1, columnspan=2, sticky="ew", padx=6, pady=4)
        ttk.Button(prediction_frame, text="Browse", command=self._browse_batch_dir).grid(row=6, column=3, padx=6, pady=4)
        batch_btn = ttk.Button(prediction_frame, text="Batch Predict", command=self._on_batch_predict)
        batch_btn.grid(row=7, column=0, columnspan=4, sticky="ew", padx=6, pady=4)
        self._action_buttons.extend([self.load_model_btn, predict_btn, batch_btn])

        class_frame = ttk.LabelFrame(left_frame, text="Class Names")
        class_frame.grid(row=2, column=0, sticky="ew", pady=6)
        class_frame.columnconfigure(1, weight=1)
        ttk.Label(class_frame, text="Dataset path:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(class_frame, textvariable=self.dataset_path_var).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(class_frame, text="Detect", command=self._detect_class_names).grid(row=0, column=2, padx=6, pady=4)

        ttk.Label(class_frame, text="Classes (comma-separated):").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(class_frame, textvariable=self.class_names_var).grid(row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=4)

        viz_frame = ttk.LabelFrame(left_frame, text="Visualization")
        viz_frame.grid(row=3, column=0, sticky="ew", pady=6)
        viz_frame.columnconfigure(0, weight=1)
        viz_buttons = [
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

    def _browse_weights(self) -> None:
        weights_dir = WEIGHTS_DIR.resolve()
        path = filedialog.askopenfilename(
            title="Select Weights File",
            initialdir=str(weights_dir),
            filetypes=[("Weights", "*.h5 *.keras *.hdf5"), ("All Files", "*.*")],
        )
        if path:
            selected = Path(path).resolve()
            if selected.parent != weights_dir:
                messagebox.showwarning("Invalid Location", "Please select a weights file from the weights folder.")
                return
            self.weights_var.set(selected.name)

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

    def _refresh_weights(self) -> None:
        weights_dir = WEIGHTS_DIR
        if not weights_dir.exists():
            weights_dir.mkdir(parents=True, exist_ok=True)

        self.weights_files = sorted([p for p in weights_dir.iterdir() if p.suffix.lower() in WEIGHT_EXTENSIONS])
        names = [p.name for p in self.weights_files]
        if self.weights_combo is not None:
            self.weights_combo.configure(values=names)
        if names:
            if not self.weights_var.get() or self.weights_var.get() not in names:
                self.weights_var.set(names[0])
            if self.load_model_btn is not None:
                self.load_model_btn.configure(state="normal")
        else:
            self.weights_var.set("")
            if self.load_model_btn is not None:
                self.load_model_btn.configure(state="disabled")
            self._log("No weights files found in the weights folder.")

    def _detect_class_names(self) -> None:
        dataset_path = Path(self.dataset_path_var.get().strip())
        if not dataset_path.exists():
            messagebox.showwarning("Missing Path", f"Dataset path does not exist: {dataset_path}")
            return
        class_names = sorted([p.name for p in dataset_path.iterdir() if p.is_dir()])
        if not class_names:
            messagebox.showwarning("No Classes", "No class folders found in the dataset path.")
            return
        self.state.class_names = class_names
        self.class_names_var.set(", ".join(class_names))
        self._log(f"Detected class names: {class_names}")

    def _resolve_class_names(self) -> List[str]:
        raw = self.class_names_var.get().strip()
        if raw:
            names = [name.strip() for name in raw.split(",") if name.strip()]
            self.state.class_names = names
            return names
        if self.state.class_names:
            return list(self.state.class_names)
        dataset_path = Path(self.dataset_path_var.get().strip())
        if dataset_path.exists():
            names = sorted([p.name for p in dataset_path.iterdir() if p.is_dir()])
            if names:
                self.state.class_names = names
                self.class_names_var.set(", ".join(names))
                return names
        return []

    def _load_model_from_weights(self, weights_path: Path) -> Optional[Any]:
        inference = self._import_optional("predict.inference", "prediction")
        if inference is None:
            return None
        model_name = weights_path.stem
        try:
            model = inference.load_model(model_name, str(weights_path))
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            self._post(messagebox.showerror, "Error", f"Failed to load model: {exc}")
            return None
        self.state.model = model
        self.state.model_name = model_name
        self.state.model_path = weights_path
        self.model_status_var.set(f"Loaded: {model_name} ({weights_path.name})")
        self._log(f"Loaded model from {weights_path}")
        return model

    def _predict_single(self, image_path: Path, top_k: int) -> Optional[Any]:
        inference = self._import_optional("predict.inference", "prediction")
        if inference is None:
            return None
        class_names = self._resolve_class_names()
        if not class_names:
            self._post(messagebox.showwarning, "Missing Classes", "Class names are required for prediction.")
            return None
        if self.state.model is None:
            self._post(messagebox.showwarning, "Missing Model", "Please load a model first.")
            return None

        try:
            result = inference.predict_single_image(self.state.model, str(image_path), class_names, top_k=top_k)
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            self._post(self._log, f"[Error] Prediction failed: {exc}")
            return None

        if "probabilities" not in result and hasattr(inference, "preprocess_input_image"):
            try:
                img_array = inference.preprocess_input_image(str(image_path), target_size=TARGET_IMAGE_SIZE)
                probs = self.state.model.predict(img_array, verbose=0)[0]
                result["probabilities"] = probs
                result["class_names"] = class_names
            except Exception as exc:
                logger.warning("Failed to compute probabilities: %s", exc)

        return result

    def _predict_batch(self, image_paths: Sequence[Path]) -> Optional[Any]:
        inference = self._import_optional("predict.inference", "prediction")
        if inference is None:
            return None
        class_names = self._resolve_class_names()
        if not class_names:
            self._post(messagebox.showwarning, "Missing Classes", "Class names are required for prediction.")
            return None
        if self.state.model is None:
            self._post(messagebox.showwarning, "Missing Model", "Please load a model first.")
            return None

        try:
            return inference.predict_batch(self.state.model, [str(p) for p in image_paths], class_names)
        except Exception as exc:
            logger.error("Batch prediction failed: %s", exc)
            self._post(self._log, f"[Error] Batch prediction failed: {exc}")
            return None

    def _on_load_model(self) -> None:
        def task() -> None:
            weights_name = self.weights_var.get().strip()
            if not weights_name:
                self._post(messagebox.showwarning, "Missing Weights", "Please select a weights file.")
                return
            weights_path = WEIGHTS_DIR / weights_name
            if not weights_path.exists():
                self._post(messagebox.showwarning, "Missing Weights", f"Weights file not found: {weights_path}")
                return
            self._load_model_from_weights(weights_path)
            self._post(messagebox.showinfo, "Model Loaded", "Model loaded successfully.")

        self._run_task("Loading model...", task)

    def _on_predict_single(self) -> None:
        def task() -> None:
            image_path = Path(self.image_path_var.get().strip())
            if not image_path.exists():
                self._post(messagebox.showwarning, "Missing Image", "Please select a valid image path.")
                return
            try:
                top_k = max(1, int(self.top_k_var.get().strip()))
            except ValueError:
                top_k = TOP_K

            self._post(self._log, f"Predicting {image_path.name}...")
            prediction = self._predict_single(image_path, top_k)
            if prediction is None:
                return

            self.state.last_prediction = prediction
            self.state.last_prediction_image = image_path
            self._post(self._log, f"Prediction result: {prediction}")

            if self.save_distribution_var.get() and isinstance(prediction, dict):
                probs = prediction.get("probabilities")
                class_names = prediction.get("class_names") or self._resolve_class_names()
                if probs is not None:
                    visualization.plot_prediction_distribution(probs, class_names, RESULTS_DIR / "last_prediction.png")
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
            batch_dir = Path(self.batch_dir_var.get().strip())
            if not batch_dir.exists():
                self._post(messagebox.showwarning, "Missing Folder", "Please select a valid batch folder.")
                return

            image_paths = collect_image_files(batch_dir)
            if not image_paths:
                self._post(messagebox.showwarning, "No Images", "No supported images found in the selected folder.")
                return

            self._post(self._log, f"Batch predicting {len(image_paths)} images...")
            predictions = self._predict_batch(image_paths)
            if predictions is None:
                return

            output_path = save_results_to_json(
                {"files": [str(p) for p in image_paths], "predictions": predictions},
                RESULTS_DIR / "batch_predictions.json",
            )
            self._post(self._log, f"Batch predictions saved to {output_path}")
            self._post(messagebox.showinfo, "Batch Prediction", "Batch predictions saved.")

        self._run_task("Batch prediction...", task)

    def _on_plot_prediction_distribution(self) -> None:
        prediction = self.state.last_prediction
        if not prediction:
            messagebox.showwarning("Missing Data", "No prediction results available.")
            return
        if isinstance(prediction, dict):
            probs = prediction.get("probabilities")
            class_names = prediction.get("class_names") or self._resolve_class_names()
        else:
            probs = None
            class_names = self._resolve_class_names()
        if probs is None:
            messagebox.showwarning("Missing Data", "Prediction probabilities not available.")
            return
        visualization.plot_prediction_distribution(probs, class_names, RESULTS_DIR / "prediction_distribution.png")
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
