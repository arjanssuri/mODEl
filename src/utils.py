"""
mODEl Utilities - Basic utility functions and session state management
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def initialize_session_state():
    """Initialize all session state variables with default values"""
    defaults = {
        'ode_system': "",
        'param_names': [],
        'initial_conditions': [],
        'datasets': {},
        'fit_results': None,
        'bootstrap_results': None,
        'batch_jobs': {},
        'active_job': None,
        'dataset_mapping': {},
        'auto_detected_vars': 1,
        'auto_detected_var_names': [],
        'bounds_code': "",
        'parsed_bounds': {},
        'parsed_initial_guesses': {},
        'trigger_model_fitting': False,
        'optimization_settings': {
            'method': 'L-BFGS-B',
            'tolerance': 1e-8,
            'max_iter': 1000,
            'multi_start': False,
            'n_starts': 10,
            'use_relative_error': True
        },
        'bootstrap_settings': {
            'n_samples': 100,
            'method': 'Residual Resampling',
            'confidence_level': 95
        },
        'visualization_settings': {
            'plot_style': 'plotly',
            'show_phase_portrait': False,
            'show_distributions': False
        }
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_completion_status() -> Dict[str, bool]:
    """Check completion status of different workflow steps"""
    return {
        'data_upload': len(st.session_state.datasets) > 0,
        'ode_definition': bool(st.session_state.ode_system and st.session_state.param_names),
        'model_fitting': st.session_state.fit_results is not None,
        'results': st.session_state.fit_results is not None,
        'bootstrap': st.session_state.bootstrap_results is not None
    } 