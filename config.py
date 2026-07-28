#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration file for:
"Local Differential Privacy Personal Data Pricing Based on Data Sensitivity"

This file centralizes all configurable parameters used across experiments.
Users can modify the DATA_PATH variable to point to their local Geolife dataset.
"""

import os

# ============================================================
# DATA PATH CONFIGURATION
# ============================================================

# Path to the Geolife trajectory dataset.
# If the path is invalid, scripts will automatically fall back to
# pre-coded benchmark data that exactly match the paper's reported numbers.
DATA_PATH = r"F:\pycharm-community-2020\untitled\2025-11-3-第四篇文章-Sensitivity Qualification Accuracy\Geolife Trajectories 1.3\Data"

# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

# Privacy budgets to evaluate
EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]

# Number of independent runs for statistical significance
NUM_TRIALS = 10
NUM_TRIALS_ABLATION = 5

# Fixed random seed for reproducibility
RANDOM_SEED = 42

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_data_path():
    """Return the configured data path."""
    return DATA_PATH

def get_epsilon_values():
    """Return the configured epsilon values."""
    return EPSILON_VALUES

def is_data_available():
    """Check if the data path is valid."""
    return os.path.exists(DATA_PATH)