# Utilities Module

This module provides essential shared functionality for logging, visualization, and general file management across the application.

## Components

- **`visualization.py`**:
  - **Plotting**: Training history (Accuracy/Loss), Confusion Matrices, and Prediction Distributions.
  - **Overlay**: Functions to display prediction labels on images.
  - Uses `matplotlib` for generating figures.

- **`logger.py`**:
  - **Setup**: Configures the Python `logging` module.
  - **Output**: Writes logs to console (stdout) and a file (`results/app.log`).

- **`helpers.py`**:
  - **File IO**: Functions for creating directories, validating paths, and finding image files.
  - **Data**: JSON serialization/deserialization helper functions.

## Note
This module aims to be dependency-free regarding the core model logic, focusing solely on support tasks.
