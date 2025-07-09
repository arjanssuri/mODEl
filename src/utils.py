"""
mODEl Utilities - Basic utility functions and session state management
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import pickle
import hashlib


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
        },
        # Cache-related settings
        'cache_enabled': True,
        'data_version': 0,  # Increment when data changes
        'last_ode_hash': "",  # Hash of ODE system for change detection
        'last_params_hash': "",  # Hash of parameters for change detection
        # Model library settings
        'saved_models': {},
        'saved_bounds': {},
        # Initial condition settings
        'ic_method': "Manual Input",
        'auto_ic_selections': {},
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Initialize model library
    from model_library import initialize_model_library
    initialize_model_library()


def get_completion_status() -> Dict[str, bool]:
    """Check completion status of different workflow steps"""
    return {
        'data_upload': len(st.session_state.datasets) > 0,
        'ode_definition': bool(st.session_state.ode_system and st.session_state.param_names),
        'model_fitting': st.session_state.fit_results is not None,
        'results': st.session_state.fit_results is not None,
        'bootstrap': st.session_state.bootstrap_results is not None
    }


@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_uploaded_data(file_content: bytes, file_name: str, delimiter: str = None) -> pd.DataFrame:
    """Cache data processing to avoid reprocessing the same file"""
    import io
    
    # Create file-like object from bytes
    file_obj = io.BytesIO(file_content)
    
    try:
        if file_name.endswith('.csv'):
            return pd.read_csv(file_obj)
        else:
            # Try different delimiters for TXT files
            file_obj.seek(0)
            content = file_obj.read().decode('utf-8')
            file_obj = io.StringIO(content)
            
            if delimiter:
                return pd.read_csv(file_obj, delimiter=delimiter)
            elif '\t' in content:
                return pd.read_csv(file_obj, delimiter='\t')
            elif ',' in content:
                return pd.read_csv(file_obj, delimiter=',')
            elif ';' in content:
                return pd.read_csv(file_obj, delimiter=';')
            elif ' ' in content:
                return pd.read_csv(file_obj, delimiter=r'\s+', engine='python')
            else:
                return pd.read_csv(file_obj, delimiter='\t')
    except Exception as e:
        st.error(f"Error processing file {file_name}: {str(e)}")
        return pd.DataFrame()


@st.cache_data
def validate_and_clean_data(data: pd.DataFrame) -> Tuple[bool, str, str, pd.DataFrame]:
    """Cache data validation and cleaning"""
    # Clean column names
    data = data.copy()
    data.columns = data.columns.str.strip()
    
    # Find time and value columns
    time_col = None
    value_col = None
    
    for col in data.columns:
        if col.lower() in ['time', 't', 'times']:
            time_col = col
            break
    
    for col in data.columns:
        if col.lower() in ['value', 'val', 'values', 'concentration', 'conc', 'amount']:
            value_col = col
            break
    
    if time_col is None or value_col is None:
        return False, "", "", data
    
    # Standardize column names
    if time_col != 'time':
        data = data.rename(columns={time_col: 'time'})
    if value_col != 'value':
        data = data.rename(columns={value_col: 'value'})
    
    # Validate data types
    data['time'] = pd.to_numeric(data['time'], errors='coerce')
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    
    # Remove rows with NaN values
    data = data.dropna()
    
    return True, time_col, value_col, data


def create_data_hash(data: Dict) -> str:
    """Create a hash of the current datasets for cache invalidation"""
    try:
        # Create a string representation of all datasets
        data_str = ""
        for name, df in data.items():
            data_str += f"{name}_{len(df)}_{df['time'].sum()}_{df['value'].sum()}"
        return hashlib.md5(data_str.encode()).hexdigest()
    except:
        return ""


def create_ode_hash(ode_system: str, param_names: List[str], initial_conditions: List[float]) -> str:
    """Create a hash of the ODE system for cache invalidation"""
    try:
        combined = f"{ode_system}_{param_names}_{initial_conditions}"
        return hashlib.md5(combined.encode()).hexdigest()
    except:
        return ""


def invalidate_cache_if_needed():
    """Check if cache should be invalidated due to data changes"""
    current_data_hash = create_data_hash(st.session_state.datasets)
    current_ode_hash = create_ode_hash(
        st.session_state.ode_system,
        st.session_state.param_names,
        st.session_state.initial_conditions
    )
    
    # Check if data has changed
    if (current_data_hash != st.session_state.get('last_data_hash', '') or
        current_ode_hash != st.session_state.get('last_ode_hash', '')):
        
        # Clear results that depend on data/ODE
        if 'fit_results' in st.session_state:
            st.session_state.fit_results = None
        if 'bootstrap_results' in st.session_state:
            st.session_state.bootstrap_results = None
        
        # Update hashes
        st.session_state.last_data_hash = current_data_hash
        st.session_state.last_ode_hash = current_ode_hash
        st.session_state.data_version += 1


@st.cache_resource
def get_cached_optimization_settings():
    """Cache optimization settings to avoid recreating them"""
    return {
        'method': 'L-BFGS-B',
        'tolerance': 1e-8,
        'max_iter': 1000,
        'multi_start': False,
        'n_starts': 10,
        'use_relative_error': True
    }


def save_session_to_browser_storage():
    """Save critical session state to browser storage (via session state persistence)"""
    # Streamlit automatically handles session state persistence
    # We just need to ensure critical data is in session state
    
    # Mark that data has been saved
    st.session_state.session_saved = True


def restore_session_from_browser_storage():
    """Restore session state from browser storage"""
    # Streamlit automatically handles this
    # Just ensure we have the required keys
    if 'session_saved' not in st.session_state:
        initialize_session_state()


def display_cache_status():
    """Display current cache status for debugging"""
    if st.sidebar.checkbox("🔧 Show Cache Status", value=False, key="show_cache_status"):
        with st.sidebar.expander("Cache Information"):
            st.write(f"**Data version:** {st.session_state.get('data_version', 0)}")
            st.write(f"**Cache enabled:** {st.session_state.get('cache_enabled', True)}")
            st.write(f"**Datasets loaded:** {len(st.session_state.datasets)}")
            st.write(f"**Results cached:** {'Yes' if st.session_state.fit_results else 'No'}")
            st.write(f"**Bootstrap cached:** {'Yes' if st.session_state.bootstrap_results else 'No'}")
            st.write(f"**Saved models:** {len(st.session_state.get('saved_models', {}))}")
            st.write(f"**Saved bounds:** {len(st.session_state.get('saved_bounds', {}))}")
            
            if st.button("🗑️ Clear All Cache", key="clear_cache_btn"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared!")
                st.rerun() 