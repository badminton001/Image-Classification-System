"""
Entry point for the Image Classification Enhancement System.
"""
from __future__ import annotations

import traceback
from pathlib import Path

from config import DATASET_PATH
from ui.interface import launch_gui
from utils.helpers import create_directories, print_section
from utils.logger import setup_logger


def _print_welcome() -> None:
    print_section("Image Classification Enhancement System")
    print(f"Default dataset path: {Path(DATASET_PATH).resolve()}")


def _check_environment(logger) -> None:
    """
    Perform lightweight environment checks so users get early feedback.
    """
    data_path = Path(DATASET_PATH)
    if not data_path.exists():
        logger.warning("Dataset path not found: %s", data_path)
        print(f"[Warning] Dataset path does not exist: {data_path}")
    else:
        logger.info("Dataset path available: %s", data_path.resolve())


def _goodbye() -> None:
    print("Thanks for using the app. Goodbye!")


def run() -> None:
    logger = setup_logger("ImageClassifier")
    logger.info("Program started.")
    create_directories()
    _print_welcome()
    _check_environment(logger)
    try:
        launch_gui()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        logger.warning("Program interrupted by user (KeyboardInterrupt).")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[Error] Unhandled exception: {exc}")
        logger.error("Unhandled exception: %s", exc)
        traceback.print_exc()
    finally:
        _goodbye()
        logger.info("Program exited.")


if __name__ == "__main__":
    run()
