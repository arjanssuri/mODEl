import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from scipy.optimize import minimize, differential_evolution
import io
import re
import os
from typing import List, Dict, Tuple, Callable
import plotly.graph_objects as go
import plotly.express as px
from ode_examples import ODE_EXAMPLES, calculate_sensitivity_matrix, estimate_parameter_confidence
import zipfile
from datetime import datetime
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="mODEl - ODE Model Fitting by Dobrovolny Lab TCU",
    page_icon="🧮",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0066CC !important;
        color: white !important;
        border-radius: 10px;
        padding: 10px 20px;
        border: none !important;
        font-weight: bold !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #0052A3 !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0080FF !important;
        color: white !important;
    }
    .parameter-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .bootstrap-warning {
        background-color: #d1ecf1;
        border: 1px solid #b8daff;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        color: #0c5460;
    }
    .bounds-code-area {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .completion-indicator {
        color: #28a745;
        font-weight: bold;
    }
    /* Regular buttons remain unchanged */
    .stButton > button {
        background-color: #0066CC !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        font-weight: bold !important;
    }
    .stButton > button:hover {
        background-color: #0052A3 !important;
        color: white !important;
    }
    /* Primary buttons - brighter blue */
    .stButton > button[kind="primary"] {
        background-color: #0080FF !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0066CC !important;
        color: white !important;
    }
    /* Secondary buttons - lighter blue */
    .stButton > button[kind="secondary"] {
        background-color: #4DA6FF !important;
        color: white !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #3399FF !important;
        color: white !important;
    }
    /* Sidebar button styling */
    .sidebar-fit-button {
        background-color: #28a745 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        font-size: 14px !important;
        padding: 10px 15px !important;
        margin: 10px 0 !important;
        width: 100% !important;
    }
    .sidebar-fit-button:hover {
        background-color: #218838 !important;
        color: white !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .keyboard-shortcut-info {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 8px;
        margin: 5px 0;
        font-size: 12px;
        color: #6c757d;
    }
</style>

<script>
// Keyboard shortcut for model fitting (Cmd + . or Ctrl + .)
document.addEventListener('keydown', function(event) {
    // Check for Cmd + . (Mac) or Ctrl + . (Windows/Linux)
    if ((event.metaKey || event.ctrlKey) && event.key === '.') {
        event.preventDefault();
        
        // Find the sidebar model fitting button by its unique key
        const buttons = document.querySelectorAll('button');
        let sidebarButton = null;
        
        buttons.forEach(button => {
            if (button.textContent.includes('Quick Model Fitting') || 
                button.getAttribute('title') === 'Run model fitting with current settings (Shortcut: Cmd/Ctrl + .)') {
                sidebarButton = button;
            }
        });
        
        if (sidebarButton) {
            sidebarButton.click();
            
            // Show visual feedback
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background-color: #28a745;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                z-index: 10000;
                font-weight: bold;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            `;
            notification.textContent = '🚀 Model fitting triggered via keyboard shortcut!';
            document.body.appendChild(notification);
            
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 3000);
        } else {
            // Show error notification if button not found
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background-color: #dc3545;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                z-index: 10000;
                font-weight: bold;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            `;
            notification.textContent = '⚠️ Quick fitting button not found. Please use the sidebar button.';
            document.body.appendChild(notification);
            
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 3000);
        }
        
        return false;
    }
});

// Add keyboard shortcut hint to page
document.addEventListener('DOMContentLoaded', function() {
    // Add global styles for keyboard shortcut hints
    const style = document.createElement('style');
    style.textContent = `
        .shortcut-tooltip {
            position: relative;
        }
        .shortcut-tooltip:hover::after {
            content: "Keyboard shortcut: Cmd + . (Mac) or Ctrl + . (Windows/Linux)";
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #333;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            z-index: 1000;
        }
        
        kbd {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            padding: 2px 4px;
            font-size: 11px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }
    `;
    document.head.appendChild(style);
});
</script>
""", unsafe_allow_html=True)

# Title and description
st.title("🧮 mODEl: Advanced ODE Model Fitting")
st.markdown("**by Dobrovolny Lab, Texas Christian University**")
st.markdown("""
**mODEl** provides comprehensive ODE modeling capabilities including:
- Multi-dataset upload and fitting
- Bootstrap analysis for parameter uncertainty
- Advanced visualization and result export
- Support for complex multi-variable systems

*Developed by the Dobrovolny Laboratory at Texas Christian University for mathematical modeling in biological systems.*
""")

# Initialize session state
if 'ode_system' not in st.session_state:
    st.session_state.ode_system = ""
if 'param_names' not in st.session_state:
    st.session_state.param_names = []
if 'initial_conditions' not in st.session_state:
    st.session_state.initial_conditions = []
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'fit_results' not in st.session_state:
    st.session_state.fit_results = None
if 'bootstrap_results' not in st.session_state:
    st.session_state.bootstrap_results = None
# New session state for batch processing
if 'batch_jobs' not in st.session_state:
    st.session_state.batch_jobs = {}
if 'active_job' not in st.session_state:
    st.session_state.active_job = None
# Enhanced state management for UI stability
if 'dataset_mapping' not in st.session_state:
    st.session_state.dataset_mapping = {}
if 'auto_detected_vars' not in st.session_state:
    st.session_state.auto_detected_vars = 1
if 'auto_detected_var_names' not in st.session_state:
    st.session_state.auto_detected_var_names = []
# Advanced analytics state preservation
if 'bounds_code' not in st.session_state:
    st.session_state.bounds_code = ""
if 'parsed_bounds' not in st.session_state:
    st.session_state.parsed_bounds = {}
if 'parsed_initial_guesses' not in st.session_state:
    st.session_state.parsed_initial_guesses = {}
if 'optimization_settings' not in st.session_state:
    st.session_state.optimization_settings = {
        'method': 'L-BFGS-B',
        'tolerance': 1e-8,
        'max_iter': 1000,
        'multi_start': False,
        'n_starts': 10,
        'use_relative_error': True
    }
if 'bootstrap_settings' not in st.session_state:
    st.session_state.bootstrap_settings = {
        'n_samples': 100,
        'method': 'Residual Resampling',
        'confidence_level': 95
    }
if 'visualization_settings' not in st.session_state:
    st.session_state.visualization_settings = {
        'plot_style': 'plotly',
        'show_phase_portrait': False,
        'show_distributions': False
    }
# Trigger for model fitting from sidebar/keyboard
if 'trigger_model_fitting' not in st.session_state:
    st.session_state.trigger_model_fitting = False

# Reusable model fitting function
def run_model_fitting():
    """Run model fitting with current settings - callable from anywhere"""
    
    # Validation checks
    if not st.session_state.param_names:
        st.error("❌ Please define your ODE system and parameters first!")
        return False
    
    if not st.session_state.datasets:
        st.error("❌ Please upload datasets first!")
        return False
    
    if not st.session_state.initial_conditions:
        st.error("❌ Please set initial conditions first!")
        return False
    
    if not st.session_state.dataset_mapping:
        st.error("❌ Please map your datasets to state variables first!")
        return False
    
    # Get current bounds and initial guesses
    bounds = {}
    initial_guesses = {}
    
    # Use parsed bounds if available, otherwise create default bounds
    if hasattr(st.session_state, 'parsed_bounds') and st.session_state.parsed_bounds:
        bounds = st.session_state.parsed_bounds
        initial_guesses = st.session_state.parsed_initial_guesses
    else:
        # Create default bounds around reasonable values
        for param in st.session_state.param_names:
            bounds[param] = (1e-6, 10.0)
            initial_guesses[param] = 1.0
    
    # Default dataset weights
    dataset_weights = {name: 1.0 for name in st.session_state.datasets.keys()}
    
    try:
        with st.spinner("🚀 Running mODEl model fitting..."):
            # Create ODE function
            def create_ode_func(param_names, ode_code):
                lines = ode_code.strip().split('\n')
                indented_lines = []
                for line in lines:
                    if line.strip():
                        indented_lines.append('    ' + line.strip())
                    else:
                        indented_lines.append('')
                
                indented_code = '\n'.join(indented_lines)
                
                func_code = f"""
def ode_system(y, t, {', '.join(param_names)}):
{indented_code}
"""
                exec(func_code, globals())
                return globals()['ode_system']
            
            ode_func = create_ode_func(st.session_state.param_names, st.session_state.ode_system)
            
            # Test ODE function
            test_params = [initial_guesses[param] for param in st.session_state.param_names]
            test_result = ode_func(st.session_state.initial_conditions, 0, *test_params)
            
            if len(test_result) != len(st.session_state.initial_conditions):
                st.error(f"❌ ODE system mismatch: Your ODE returns {len(test_result)} derivatives but you have {len(st.session_state.initial_conditions)} initial conditions.")
                return False
            
            # Prepare all datasets
            all_times = []
            all_data = []
            for dataset_name, data in st.session_state.datasets.items():
                all_times.extend(data['time'].values)
                all_data.append(data)
            
            # Get unique sorted time points
            unique_times = sorted(set(all_times))
            t_data = np.array(unique_times)
            
            # Multi-objective optimization function
            def objective(params):
                try:
                    # Solve ODE
                    sol = odeint(ode_func, st.session_state.initial_conditions, t_data, 
                               args=tuple(params))
                    
                    total_ssr = 0
                    for dataset_name, data in st.session_state.datasets.items():
                        var_idx = st.session_state.dataset_mapping[dataset_name]
                        
                        # Interpolate model solution to data time points
                        model_vals = np.interp(data['time'], t_data, sol[:, var_idx])
                        
                        # Calculate error
                        if st.session_state.optimization_settings['use_relative_error']:
                            error = ((model_vals - data['value']) / (np.abs(data['value']) + 1e-10))**2
                        else:
                            error = (model_vals - data['value'])**2
                        
                        # Weight by dataset
                        ssr = np.sum(error) * dataset_weights[dataset_name]
                        total_ssr += ssr
                    
                    return total_ssr
                except:
                    return 1e12
            
            # Setup optimization
            opt_bounds = [bounds[param] for param in st.session_state.param_names]
            x0 = [initial_guesses[param] for param in st.session_state.param_names]
            
            # Run optimization
            if st.session_state.optimization_settings['multi_start']:
                best_result = None
                best_cost = np.inf
                
                for i in range(st.session_state.optimization_settings['n_starts']):
                    # Random initial point
                    x0_random = []
                    for (low, high) in opt_bounds:
                        if np.isfinite(low) and np.isfinite(high):
                            x0_random.append(np.random.uniform(low, high))
                        else:
                            x0_random.append(np.random.lognormal(0, 1))
                    
                    result = minimize(objective, x0_random, method=st.session_state.optimization_settings['method'], 
                                    bounds=opt_bounds, options={'maxiter': st.session_state.optimization_settings['max_iter']})
                    
                    if result.fun < best_cost:
                        best_result = result
                        best_cost = result.fun
                
                result = best_result
            else:
                result = minimize(objective, x0, method=st.session_state.optimization_settings['method'], 
                                bounds=opt_bounds, options={'maxiter': st.session_state.optimization_settings['max_iter']})
            
            # Store results
            st.session_state.fit_results = {
                'params': dict(zip(st.session_state.param_names, result.x)),
                'cost': result.fun,
                'success': result.success,
                'message': result.message,
                'result_obj': result,
                'dataset_weights': dataset_weights,
                'fitting_options': {
                    'use_relative_error': st.session_state.optimization_settings['use_relative_error'],
                    'use_log_transform': False,
                    'normalize_by_initial': False
                }
            }
            
            # Success message with parameter summary
            param_summary = ", ".join([f"{param}={value:.3e}" for param, value in st.session_state.fit_results['params'].items()])
            st.success(f"✅ mODEl fitting completed! Cost: {result.fun:.3e}")
            st.info(f"📊 **Fitted Parameters:** {param_summary}")
            
            return True
    
    except Exception as e:
        st.error(f"❌ Model fitting error: {str(e)}")
        return False

# Function to check completion status
def get_completion_status():
    return {
        'data_upload': len(st.session_state.datasets) > 0,
        'ode_definition': bool(st.session_state.ode_system and st.session_state.param_names),
        'model_fitting': st.session_state.fit_results is not None,
        'results': st.session_state.fit_results is not None,
        'bootstrap': st.session_state.bootstrap_results is not None
    }

# Get completion status
completion = get_completion_status()

# Create tab labels with completion indicators
tab_labels = [
    f"📁 Data Upload{'✅' if completion['data_upload'] else ''}",
    f"🧬 ODE Definition{'✅' if completion['ode_definition'] else ''}",
    f"📊 Model Fitting{'✅' if completion['model_fitting'] else ''}",
    f"📈 Results{'✅' if completion['results'] else ''}",
    f"🎯 Bootstrap Analysis{'✅' if completion['bootstrap'] else ''}",
    "📚 Examples"
]

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_labels)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Optimization method selection
    opt_method = st.selectbox(
        "Optimization Method",
        ["L-BFGS-B", "Nelder-Mead", "SLSQP", "Powell", "TNC", "Differential Evolution"],
        index=["L-BFGS-B", "Nelder-Mead", "SLSQP", "Powell", "TNC", "Differential Evolution"].index(st.session_state.optimization_settings['method']),
        help="Select the optimization algorithm for parameter fitting"
    )
    st.session_state.optimization_settings['method'] = opt_method
    
    # Tolerance settings
    st.subheader("Convergence Settings")
    tol = st.number_input(
        "Tolerance", 
        value=st.session_state.optimization_settings['tolerance'], 
        format="%.2e", 
        help="Convergence tolerance"
    )
    st.session_state.optimization_settings['tolerance'] = tol
    
    max_iter = st.number_input(
        "Max Iterations", 
        value=st.session_state.optimization_settings['max_iter'], 
        min_value=100, 
        step=100
    )
    st.session_state.optimization_settings['max_iter'] = max_iter
    
    # Advanced options
    st.subheader("Advanced Options")
    use_relative_error = st.checkbox(
        "Use Relative Error", 
        value=st.session_state.optimization_settings['use_relative_error'], 
        help="Use relative error instead of absolute error"
    )
    st.session_state.optimization_settings['use_relative_error'] = use_relative_error
    
    multi_start = st.checkbox(
        "Multi-start Optimization", 
        value=st.session_state.optimization_settings['multi_start']
    )
    st.session_state.optimization_settings['multi_start'] = multi_start
    
    if multi_start:
        n_starts = st.number_input(
            "Number of starts", 
            value=st.session_state.optimization_settings['n_starts'], 
            min_value=2, 
            max_value=100
        )
        st.session_state.optimization_settings['n_starts'] = n_starts
    
    # Bootstrap settings
    st.subheader("Bootstrap Analysis")
    enable_bootstrap = st.checkbox("Enable Bootstrap Analysis", value=False)
    if enable_bootstrap:
        n_bootstrap_samples = st.number_input("Number of Bootstrap Samples", 
                                                value=st.session_state.bootstrap_settings['n_samples'], min_value=10, max_value=1000)
        
        bootstrap_method = st.selectbox("Bootstrap Method", 
                                          ["Residual Resampling", "Parametric Bootstrap"],
                                          index=["Residual Resampling", "Parametric Bootstrap"].index(st.session_state.bootstrap_settings['method']))
        
        confidence_level = st.selectbox("Confidence Level", [90, 95, 99], 
                                           index=[90, 95, 99].index(st.session_state.bootstrap_settings['confidence_level']))
        
        # Update session state with current selections
        st.session_state.bootstrap_settings['n_samples'] = n_bootstrap_samples
        st.session_state.bootstrap_settings['method'] = bootstrap_method
        st.session_state.bootstrap_settings['confidence_level'] = confidence_level
    
    # Plot settings
    st.subheader("Visualization Settings")
    plot_style = st.selectbox(
        "Plot Style", 
        ["plotly", "seaborn"],
        index=["plotly", "seaborn"].index(st.session_state.visualization_settings['plot_style'])
    )
    st.session_state.visualization_settings['plot_style'] = plot_style
    
    show_phase_portrait = st.checkbox(
        "Show Phase Portrait (2D systems)", 
        value=st.session_state.visualization_settings['show_phase_portrait']
    )
    st.session_state.visualization_settings['show_phase_portrait'] = show_phase_portrait
    
    show_distributions = st.checkbox(
        "Show Parameter Distributions", 
        value=st.session_state.visualization_settings['show_distributions']
    )
    st.session_state.visualization_settings['show_distributions'] = show_distributions
    
    # Completion status display
    st.subheader("📋 Progress Tracker")
    completion = get_completion_status()
    
    progress_items = [
        ("📁 Data Upload", completion['data_upload']),
        ("🧬 ODE Definition", completion['ode_definition']),
        ("📊 Model Fitting", completion['model_fitting']),
        ("📈 Results Available", completion['results']),
        ("🎯 Bootstrap Done", completion['bootstrap'])
    ]
    
    for item, is_complete in progress_items:
        if is_complete:
            st.success(f"✅ {item}")
        else:
            st.info(f"⏳ {item}")
    
    # Overall progress
    completed_steps = sum(completion.values())
    total_steps = len(completion)
    progress_percentage = (completed_steps / total_steps) * 100
    
    st.subheader("Overall Progress")
    st.progress(progress_percentage / 100)
    st.markdown(f"**{completed_steps}/{total_steps} steps completed ({progress_percentage:.0f}%)**")

    # Quick Model Fitting from Sidebar
    st.markdown("---")
    st.subheader("🚀 Quick Actions")
    
    # Show keyboard shortcut info
    st.markdown("""
    <div class="keyboard-shortcut-info">
    💡 <strong>Tip:</strong> Press <kbd>Cmd + .</kbd> (Mac) or <kbd>Ctrl + .</kbd> (Windows/Linux) to quickly run model fitting from anywhere!
    </div>
    """, unsafe_allow_html=True)
    
    # Quick fit button with custom styling and tooltip
    if st.button("🎯 Quick Model Fitting", 
                key="sidebar_model_fitting",
                help="Run model fitting with current settings (Shortcut: Cmd/Ctrl + .)",
                type="primary"):
        st.session_state.trigger_model_fitting = True
    
    # Show current readiness status
    ready_for_fitting = (
        len(st.session_state.datasets) > 0 and 
        bool(st.session_state.ode_system and st.session_state.param_names) and
        len(st.session_state.initial_conditions) > 0 and
        len(st.session_state.dataset_mapping) > 0
    )
    
    if ready_for_fitting:
        st.success("✅ Ready for model fitting!")
        if st.session_state.param_names:
            st.info(f"📊 **Parameters to fit:** {', '.join(st.session_state.param_names)}")
        if st.session_state.datasets:
            st.info(f"📁 **Datasets:** {len(st.session_state.datasets)} loaded")
    else:
        missing_items = []
        if len(st.session_state.datasets) == 0:
            missing_items.append("Upload datasets")
        if not (st.session_state.ode_system and st.session_state.param_names):
            missing_items.append("Define ODE system")
        if len(st.session_state.initial_conditions) == 0:
            missing_items.append("Set initial conditions")
        if len(st.session_state.dataset_mapping) == 0:
            missing_items.append("Map datasets to variables")
        
        st.warning("⚠️ **Setup needed:**")
        for item in missing_items:
            st.write(f"• {item}")

# Check and handle model fitting trigger from sidebar or keyboard shortcut
if st.session_state.trigger_model_fitting:
    st.session_state.trigger_model_fitting = False  # Reset trigger
    run_model_fitting()

# Tab 1: Enhanced Data Upload with Batch Processing
with tab1:
    st.header("Upload Experimental Data to mODEl")
    
    # Data upload method selection
    upload_method = st.radio(
        "Choose upload method:",
        ["Individual Files", "Batch Folder Upload"],
        help="Upload individual files or process multiple files as batch jobs"
    )
    
    if upload_method == "Individual Files":
        # Original multi-dataset upload
    st.subheader("Multi-Dataset Upload")
    st.info("Upload multiple datasets for different variables in your ODE system. Each dataset should have 'time' and 'value' columns.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dynamic dataset upload
        uploaded_file = st.file_uploader(
            "Choose a file (txt or csv)",
            type=['txt', 'csv'],
            help="Upload experimental data with 'time' and 'value' columns for analysis in mODEl"
        )
        
        # Auto-detect dataset name from filename
        if uploaded_file is not None:
            # Extract filename without extension for default dataset name
            default_name = uploaded_file.name.rsplit('.', 1)[0]
            dataset_name = st.text_input("Dataset Name", value=default_name, placeholder="e.g., viral_load, interferon")
        else:
            dataset_name = st.text_input("Dataset Name", placeholder="e.g., viral_load, interferon")
        
        if uploaded_file is not None and dataset_name:
            try:
                # Read the file with better parsing
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    # Try different delimiters for TXT files
                    content = uploaded_file.read().decode('utf-8')
                    uploaded_file.seek(0)  # Reset file pointer
                    
                    # Detect delimiter
                    if '\t' in content:
                        data = pd.read_csv(uploaded_file, delimiter='\t')
                    elif ',' in content:
                        data = pd.read_csv(uploaded_file, delimiter=',')
                    elif ';' in content:
                        data = pd.read_csv(uploaded_file, delimiter=';')
                    elif ' ' in content:
                        data = pd.read_csv(uploaded_file, delimiter=r'\s+', engine='python')
                    else:
                        data = pd.read_csv(uploaded_file, delimiter='\t')
                
                # Clean column names (remove extra whitespace)
                data.columns = data.columns.str.strip()
                
                # Check for required columns (case insensitive)
                col_names = [col.lower() for col in data.columns]
                time_col = None
                value_col = None
                
                # Find time column
                for col in data.columns:
                    if col.lower() in ['time', 't', 'times']:
                        time_col = col
                        break
                
                # Find value column
                for col in data.columns:
                    if col.lower() in ['value', 'val', 'values', 'concentration', 'conc', 'amount']:
                        value_col = col
                        break
                
                if time_col is None or value_col is None:
                    st.error(f"Data must have 'time' and 'value' columns. Found columns: {', '.join(data.columns)}")
                    st.info("Acceptable column names:\n- Time: 'time', 't', 'times'\n- Value: 'value', 'val', 'values', 'concentration', 'conc', 'amount'")
                else:
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
                    
                    if len(data) == 0:
                        st.error("No valid data rows found after cleaning")
                    else:
                        st.session_state.datasets[dataset_name] = data
                        st.success(f"✅ Dataset '{dataset_name}' loaded successfully! ({len(data)} data points)")
                        
                        # Show column mapping info
                        if time_col != 'time' or value_col != 'value':
                            st.info(f"Column mapping: '{time_col}' → time, '{value_col}' → value")
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.info("Make sure your file has columns for time and values, separated by tabs, commas, or spaces.")
        
        # Remove dataset button
        if st.session_state.datasets:
            dataset_to_remove = st.selectbox("Remove Dataset", [""] + list(st.session_state.datasets.keys()))
            if dataset_to_remove and st.button("Remove Selected Dataset"):
                del st.session_state.datasets[dataset_to_remove]
                st.rerun()
    
    with col2:
        if st.session_state.datasets:
            st.subheader("Loaded Datasets")
            for name, data in st.session_state.datasets.items():
                st.info(f"""
                **{name}**
                - Rows: {len(data)}
                - Time range: {data['time'].min():.2f} - {data['time'].max():.2f}
                - Value range: {data['value'].min():.2f} - {data['value'].max():.2f}
                """)
    
    else:  # Batch Folder Upload
        st.subheader("📁 Batch Folder Upload")
        st.info("""
        Upload multiple files at once to create separate analysis jobs. Each job can contain multiple related datasets.
        - Upload multiple files containing experimental data
        - Each folder/group becomes a separate modeling job
        - Process jobs independently with different ODE systems and parameters
        """)
        
        # Multiple file uploader for batch processing
        uploaded_files = st.file_uploader(
            "Choose multiple files (txt or csv)",
            type=['txt', 'csv'],
            accept_multiple_files=True,
            help="Upload multiple data files to create batch jobs for analysis in mODEl"
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} files uploaded**")
            
            # Group files by prefix or organize them
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Job Organization")
                job_organization = st.radio(
                    "How to organize files into jobs:",
                    ["Create separate job for each file", "Group by filename prefix", "Group all files into one job"],
                    help="Choose how to organize the uploaded files into modeling jobs"
                )
                
                # Job naming and organization
                if job_organization == "Create separate job for each file":
                    # Each file becomes its own job
                    organized_jobs = {}
                    for file in uploaded_files:
                        job_name = file.name.rsplit('.', 1)[0]  # Remove extension
                        organized_jobs[job_name] = [file]
                
                elif job_organization == "Group by filename prefix":
                    # Group files by common prefix (before underscore or dash)
                    organized_jobs = {}
                    for file in uploaded_files:
                        # Extract prefix (before first underscore or dash)
                        base_name = file.name.rsplit('.', 1)[0]
                        if '_' in base_name:
                            prefix = base_name.split('_')[0]
                        elif '-' in base_name:
                            prefix = base_name.split('-')[0]
            else:
                            prefix = base_name
                        
                        if prefix not in organized_jobs:
                            organized_jobs[prefix] = []
                        organized_jobs[prefix].append(file)
                
                else:  # Group all files into one job
                    organized_jobs = {"batch_job": uploaded_files}
                
                # Display organization
                st.write("**Proposed Job Organization:**")
                for job_name, files in organized_jobs.items():
                    st.write(f"- **{job_name}**: {len(files)} files")
                    for file in files:
                        st.write(f"  - {file.name}")
            
            with col2:
                st.subheader("Process Batch Jobs")
                
                if st.button("🚀 Create Batch Jobs", type="primary"):
                    # Process each job
                    jobs_created = 0
                    jobs_failed = 0
                    
                    for job_name, files in organized_jobs.items():
                        try:
                            # Process files for this job
                            job_datasets = {}
                            
                            for file in files:
                                # Read and process each file
                                try:
                                    if file.name.endswith('.csv'):
                                        data = pd.read_csv(file)
        else:
                                        # Try different delimiters for TXT files
                                        content = file.read().decode('utf-8')
                                        file.seek(0)  # Reset file pointer
                                        
                                        # Detect delimiter
                                        if '\t' in content:
                                            data = pd.read_csv(file, delimiter='\t')
                                        elif ',' in content:
                                            data = pd.read_csv(file, delimiter=',')
                                        elif ';' in content:
                                            data = pd.read_csv(file, delimiter=';')
                                        elif ' ' in content:
                                            data = pd.read_csv(file, delimiter=r'\s+', engine='python')
            else:
                                            data = pd.read_csv(file, delimiter='\t')
                                    
                                    # Clean column names
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
                                    
                                    if time_col and value_col:
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
                                        
                                        if len(data) > 0:
                                            dataset_name = file.name.rsplit('.', 1)[0]
                                            job_datasets[dataset_name] = data
                                
                                except Exception as e:
                                    st.warning(f"Could not process file {file.name}: {str(e)}")
                            
                            # Create job if datasets were successfully processed
                            if job_datasets:
                                st.session_state.batch_jobs[job_name] = {
                                    'datasets': job_datasets,
                                    'status': 'ready',
                                    'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    # Core ODE settings
                                    'ode_system': '',
                                    'param_names': [],
                                    'initial_conditions': [],
                                    'dataset_mapping': {},
                                    'auto_detected_vars': 1,
                                    'auto_detected_var_names': [],
                                    # Advanced analytics settings
                                    'bounds_code': '',
                                    'parsed_bounds': {},
                                    'parsed_initial_guesses': {},
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
                                    # Results
                                    'fit_results': None,
                                    'bootstrap_results': None
                                }
                                jobs_created += 1
                            else:
                                jobs_failed += 1
                        
                        except Exception as e:
                            st.error(f"Error creating job {job_name}: {str(e)}")
                            jobs_failed += 1
                    
                    if jobs_created > 0:
                        st.success(f"✅ Successfully created {jobs_created} batch jobs!")
                        if jobs_failed > 0:
                            st.warning(f"⚠️ {jobs_failed} jobs failed to create")
                        st.rerun()
                    else:
                        st.error("❌ No jobs could be created. Check your file formats.")
        
        # Display existing batch jobs
        if st.session_state.batch_jobs:
            st.subheader("📋 Batch Jobs Management")
            
            # Job selection and management
            col1, col2 = st.columns([2, 1])
                
                with col1:
                # Job selector
                job_names = list(st.session_state.batch_jobs.keys())
                selected_job = st.selectbox(
                    "Select job to work on:",
                    [""] + job_names,
                    help="Select a batch job to work on. This will load its datasets into the main workspace."
                )
                
                if selected_job:
                    job_data = st.session_state.batch_jobs[selected_job]
                    
                    st.write(f"**Job: {selected_job}**")
                    st.write(f"- Status: {job_data['status']}")
                    st.write(f"- Created: {job_data['created']}")
                    st.write(f"- Datasets: {len(job_data['datasets'])}")
                    
                    # List datasets in this job
                    for dataset_name, data in job_data['datasets'].items():
                        st.write(f"  - {dataset_name}: {len(data)} points")
                    
                    col_load, col_delete = st.columns(2)
                    
                    with col_load:
                        if st.button(f"🔄 Load Job '{selected_job}'", type="primary"):
                            # Load job data into main workspace
                            st.session_state.datasets = job_data['datasets'].copy()
                            st.session_state.active_job = selected_job
                            
                            # Load job's core ODE system and parameters
                            st.session_state.ode_system = job_data.get('ode_system', '')
                            st.session_state.param_names = job_data.get('param_names', [])
                            st.session_state.initial_conditions = job_data.get('initial_conditions', [])
                            st.session_state.dataset_mapping = job_data.get('dataset_mapping', {})
                            st.session_state.auto_detected_vars = job_data.get('auto_detected_vars', 1)
                            st.session_state.auto_detected_var_names = job_data.get('auto_detected_var_names', [])
                            
                            # Load advanced analytics settings
                            st.session_state.bounds_code = job_data.get('bounds_code', '')
                            st.session_state.parsed_bounds = job_data.get('parsed_bounds', {})
                            st.session_state.parsed_initial_guesses = job_data.get('parsed_initial_guesses', {})
                            
                            # Load optimization settings
                            if 'optimization_settings' in job_data:
                                st.session_state.optimization_settings.update(job_data['optimization_settings'])
                            
                            # Load bootstrap settings
                            if 'bootstrap_settings' in job_data:
                                st.session_state.bootstrap_settings.update(job_data['bootstrap_settings'])
                            
                            # Load visualization settings
                            if 'visualization_settings' in job_data:
                                st.session_state.visualization_settings.update(job_data['visualization_settings'])
                            
                            # Load results
                            st.session_state.fit_results = job_data.get('fit_results')
                            st.session_state.bootstrap_results = job_data.get('bootstrap_results')
                            
                            st.success(f"✅ Loaded job '{selected_job}' with all advanced analytics settings!")
                            st.rerun()
                    
                    with col_delete:
                        if st.button(f"🗑️ Delete Job", key=f"delete_{selected_job}"):
                            del st.session_state.batch_jobs[selected_job]
                            if st.session_state.active_job == selected_job:
                                st.session_state.active_job = None
                            st.rerun()
            
            with col2:
                # Current active job display
                if st.session_state.active_job:
                    st.info(f"**Active Job:** {st.session_state.active_job}")
                    
                    if st.button("💾 Save Current State to Job"):
                        # Save current workspace state back to the active job
                        job_data = st.session_state.batch_jobs[st.session_state.active_job]
                        
                        # Save core ODE settings
                        job_data['ode_system'] = st.session_state.ode_system
                        job_data['param_names'] = st.session_state.param_names
                        job_data['initial_conditions'] = st.session_state.initial_conditions
                        job_data['dataset_mapping'] = st.session_state.dataset_mapping
                        job_data['auto_detected_vars'] = st.session_state.auto_detected_vars
                        job_data['auto_detected_var_names'] = st.session_state.auto_detected_var_names
                        
                        # Save advanced analytics settings
                        job_data['bounds_code'] = st.session_state.bounds_code
                        job_data['parsed_bounds'] = st.session_state.parsed_bounds
                        job_data['parsed_initial_guesses'] = st.session_state.parsed_initial_guesses
                        job_data['optimization_settings'] = st.session_state.optimization_settings.copy()
                        job_data['bootstrap_settings'] = st.session_state.bootstrap_settings.copy()
                        job_data['visualization_settings'] = st.session_state.visualization_settings.copy()
                        
                        # Save results
                        job_data['fit_results'] = st.session_state.fit_results
                        job_data['bootstrap_results'] = st.session_state.bootstrap_results
                        
                        # Update status
                        job_data['status'] = 'configured' if st.session_state.ode_system else 'ready'
                        
                        st.success(f"✅ Saved current state with all advanced settings to job '{st.session_state.active_job}'")
        else:
                    st.warning("No active job selected")
            
            # Jobs summary table
            st.subheader("📊 Jobs Summary")
            
            job_summary = []
            for job_name, job_data in st.session_state.batch_jobs.items():
                job_summary.append({
                    'Job Name': job_name,
                    'Status': job_data['status'],
                    'Datasets': len(job_data['datasets']),
                    'ODE Defined': '✅' if job_data['ode_system'] else '❌',
                    'Fitted': '✅' if job_data['fit_results'] else '❌',
                    'Bootstrap': '✅' if job_data['bootstrap_results'] else '❌',
                    'Created': job_data['created']
                })
            
            if job_summary:
                summary_df = pd.DataFrame(job_summary)
                st.dataframe(summary_df, use_container_width=True)
                
                # Export jobs summary
                if st.button("📥 Export Jobs Summary"):
                    csv_buffer = io.StringIO()
                    summary_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="Download Jobs Summary CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"mODEl_batch_jobs_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

# Tab 2: ODE Definition (updated for multi-dataset)
with tab2:
    st.header("Define Your ODE System")
    
    # Add example selector
    use_example = st.checkbox("Use an example ODE system")
    
    if use_example:
        selected_example = st.selectbox(
            "Select an ODE system:",
            list(ODE_EXAMPLES.keys())
        )
        
        if selected_example:
            example = ODE_EXAMPLES[selected_example]
            st.info(f"**{selected_example}**: {example['description']}")
            
            if st.button("Load Example"):
                st.session_state.ode_system = example['code']
                st.session_state.param_names = example['parameters']
                st.rerun()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("ODE System")
        st.info("""
        Define your ODE system using Python syntax.
        - Use `y[0], y[1], ...` for state variables
        - Use parameter names directly (e.g., `beta`, `gamma`)
        - Return derivatives as a list
        
        **Multi-variable Example:**
        ```python
        # Viral dynamics with immune response
        T, R, I, V, F = y
        dTdt = -beta * T * V - gamma * T * F
        dRdt = gamma * T * F - rho * R
        dIdt = beta * T * V - delta * I
        dVdt = p * I - c * V
        dFdt = I - alpha * F
        return [dTdt, dRdt, dIdt, dVdt, dFdt]
        ```
        """)
        
        ode_code = st.text_area(
            "Enter ODE system:",
            value=st.session_state.ode_system,
            height=250,
            placeholder="# Define your ODE system here\n# Example:\n# dxdt = -k * x\n# return [dxdt]"
        )
        
        if ode_code:
            st.session_state.ode_system = ode_code
            
            # Auto-detect state variables from ODE code
            def detect_state_variables(ode_code):
                """Detect the number of state variables from ODE code"""
                lines = ode_code.strip().split('\n')
                max_y_index = -1
                unpacked_vars = []
                
                for line in lines:
                    line = line.strip()
                    
                    # Check for variable unpacking (e.g., "T, R, I, V, F = y")
                    if '= y' in line and not line.startswith('#'):
                        # Extract variable names before = y
                        var_part = line.split('= y')[0].strip()
                        # Remove any comments
                        var_part = var_part.split('#')[0].strip()
                        # Split by comma and clean up
                        if ',' in var_part:
                            unpacked_vars = [v.strip() for v in var_part.split(',')]
                            return len(unpacked_vars), unpacked_vars
                    
                    # Check for y[i] indexing
                    import re
                    y_indices = re.findall(r'y\[(\d+)\]', line)
                    for idx_str in y_indices:
                        idx = int(idx_str)
                        max_y_index = max(max_y_index, idx)
                
                # If we found y[i] indices, return max_index + 1
                if max_y_index >= 0:
                    return max_y_index + 1, []
                
                # If no clear detection, return 1 as default
                return 1, []
            
            n_vars_detected, var_names = detect_state_variables(ode_code)
            
            # Update session state with detected variables
            st.session_state.auto_detected_vars = n_vars_detected
            st.session_state.auto_detected_var_names = var_names
            
            # Extract parameter names
            param_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            all_names = re.findall(param_pattern, ode_code)
            # Filter out common Python keywords, variables, and detected variable names
            exclude = {'y', 'dydt', 'return', 'def', 'if', 'else', 'for', 'while', 'in', 'and', 'or', 'not', 't', 'N', 
                      'dTdt', 'dRdt', 'dIdt', 'dVdt', 'dFdt', 'T', 'R', 'I', 'V', 'F', 'dxdt', 'dx', 'dt'}
            exclude.update(var_names)  # Add detected variable names to exclusion
            param_names = list(set(all_names) - exclude)
            
            if param_names:
                st.session_state.param_names = sorted(param_names)
                
                # Display detection results
                col_detect1, col_detect2 = st.columns(2)
                with col_detect1:
                    st.success(f"🔍 **Auto-detected {n_vars_detected} state variables**")
                    if var_names:
                        st.info(f"Variable names: {', '.join(var_names)}")
                    else:
                        st.info(f"Using y[0] through y[{n_vars_detected-1}]")
                
                with col_detect2:
                    st.success(f"📊 **Detected {len(param_names)} parameters**")
                    st.info(f"Parameters: {', '.join(param_names)}")
            else:
                st.warning("No parameters detected. Please check your ODE system.")
    
    with col2:
        st.subheader("Initial Conditions & Data Mapping")
        
        if st.session_state.datasets:
            # Get number of variables from auto-detection
            if st.session_state.ode_system and hasattr(st.session_state, 'auto_detected_vars'):
                n_vars = st.session_state.auto_detected_vars
                var_names = st.session_state.auto_detected_var_names
                
                # Option to override auto-detection
                with st.expander("🔧 Override Auto-Detection"):
                    st.markdown("**Auto-detection found:** " + 
                              (f"{n_vars} variables ({', '.join(var_names)})" if var_names else f"{n_vars} variables (y[0] to y[{n_vars-1}])"))
                    
                    override_detection = st.checkbox("Override auto-detection", value=False)
                    if override_detection:
                        n_vars = st.number_input("Manual number of state variables", value=n_vars, min_value=1, max_value=10)
                        st.info(f"Using manual setting: {n_vars} variables")
                
                st.write("**Initial Conditions:**")
                st.markdown(f"*Setting initial conditions for {n_vars} state variables*")
                
                # Option to set initial conditions from first data values
                with st.expander("🔧 Initial Condition Options"):
                    ic_method = st.radio(
                        "How to set initial conditions:",
                        ["Manual Input", "Use First Data Values", "Use Dataset-Specific First Values"],
                        help="Choose how to set initial conditions for state variables"
                    )
                    
                    if ic_method == "Use First Data Values":
                        st.info("""
                        **Use First Data Values**: Uses the first value from each mapped dataset as the initial condition.
                        - Automatically sets initial conditions based on actual data
                        - Requires datasets to be mapped to state variables first
                        - Good for cases where t=0 represents the actual starting point of your measurements
                        """)
                    elif ic_method == "Use Dataset-Specific First Values":
                        st.info("""
                        **Use Dataset-Specific First Values**: Like above, but allows you to choose which dataset's first value to use for each state variable.
                        - More control over which dataset provides each initial condition
                        - Useful when you have multiple datasets for related variables
                        """)
                    else:
                        st.info("""
                        **Manual Input**: Set initial conditions manually using number inputs.
                        - Full control over initial values
                        - Good for theoretical starting points that may differ from measured values
                        """)
                
                initial_conditions = []
                
                # Get stable job identifier for widget keys
                job_key = st.session_state.active_job if st.session_state.active_job else 'main'
                
                if ic_method == "Manual Input":
                    # Original manual input method with stable keys
                if n_vars <= 3:
                    # Single row for 1-3 variables
                    cols = st.columns(n_vars)
                    for i in range(n_vars):
                        with cols[i]:
                            if var_names and i < len(var_names):
                                label = f"{var_names[i]}(0)"
                                help_text = f"Initial condition for {var_names[i]}"
                            else:
                                label = f"y[{i}](0)"
                                help_text = f"Initial condition for y[{i}]"
                            
                                # Use stable key and preserve existing values
                                current_value = 0.0
                                if i < len(st.session_state.initial_conditions):
                                    current_value = st.session_state.initial_conditions[i]
                                
                                ic = st.number_input(
                                    label, 
                                    value=current_value, 
                                    key=f"manual_ic_{i}_{job_key}", 
                                    help=help_text
                                )
                            initial_conditions.append(ic)
                else:
                    # Multiple rows for >3 variables
                    for i in range(n_vars):
                        if var_names and i < len(var_names):
                            label = f"{var_names[i]}(0)"
                            help_text = f"Initial condition for {var_names[i]}"
                        else:
                            label = f"y[{i}](0)"
                            help_text = f"Initial condition for y[{i}]"
                        
                            # Use stable key and preserve existing values
                            current_value = 0.0
                            if i < len(st.session_state.initial_conditions):
                                current_value = st.session_state.initial_conditions[i]
                            
                            ic = st.number_input(
                                label, 
                                value=current_value, 
                                key=f"manual_ic_{i}_{job_key}", 
                                help=help_text
                            )
                        initial_conditions.append(ic)
                
                elif ic_method == "Use First Data Values":
                    # Automatically use first values from mapped datasets
                    if hasattr(st.session_state, 'dataset_mapping') and st.session_state.dataset_mapping:
                        # Create initial conditions based on dataset mapping
                        for i in range(n_vars):
                            # Find which dataset maps to this state variable
                            mapped_dataset = None
                            for dataset_name, var_idx in st.session_state.dataset_mapping.items():
                                if var_idx == i:
                                    mapped_dataset = dataset_name
                                    break
                            
                            if mapped_dataset and mapped_dataset in st.session_state.datasets:
                                # Use first value from the mapped dataset
                                first_value = st.session_state.datasets[mapped_dataset]['value'].iloc[0]
                                initial_conditions.append(first_value)
                                
                                # Display the auto-set value
                                if var_names and i < len(var_names):
                                    st.info(f"{var_names[i]}(0) = {first_value:.4f} (from {mapped_dataset})")
                                else:
                                    st.info(f"y[{i}](0) = {first_value:.4f} (from {mapped_dataset})")
                            else:
                                # No dataset mapped to this variable, use default
                                initial_conditions.append(0.0)
                                if var_names and i < len(var_names):
                                    st.warning(f"{var_names[i]}(0) = 0.0 (no dataset mapped)")
                                else:
                                    st.warning(f"y[{i}](0) = 0.0 (no dataset mapped)")
                    else:
                        st.warning("⚠️ No dataset mapping found. Please map datasets to state variables first, or use manual input.")
                        # Fall back to manual input
                        for i in range(n_vars):
                            initial_conditions.append(0.0)
                
                elif ic_method == "Use Dataset-Specific First Values":
                    # Allow user to select which dataset to use for each variable
                    dataset_names = list(st.session_state.datasets.keys())
                    if dataset_names:
                        st.write("**Select dataset for each initial condition:**")
                        
                        for i in range(n_vars):
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                if var_names and i < len(var_names):
                                    var_label = f"{var_names[i]}(0)"
                                else:
                                    var_label = f"y[{i}](0)"
                                
                                # Dropdown to select dataset with stable key
                                selected_dataset = st.selectbox(
                                    f"Dataset for {var_label}:",
                                    ["Manual"] + dataset_names,
                                    key=f"ic_dataset_select_{i}_{job_key}",
                                    help=f"Choose which dataset's first value to use for {var_label}"
                                )
                            
                            with col2:
                                if selected_dataset == "Manual":
                                    # Manual input with stable key
                                    current_value = 0.0
                                    if i < len(st.session_state.initial_conditions):
                                        current_value = st.session_state.initial_conditions[i]
                                    
                                    ic = st.number_input(
                                        f"{var_label} value:",
                                        value=current_value,
                                        key=f"ic_dataset_manual_{i}_{job_key}",
                                        help=f"Manual initial condition for {var_label}"
                                    )
                                    initial_conditions.append(ic)
                                else:
                                    # Use first value from selected dataset
                                    first_value = st.session_state.datasets[selected_dataset]['value'].iloc[0]
                                    initial_conditions.append(first_value)
                                    st.info(f"{var_label} = {first_value:.4f}")
                    else:
                        st.warning("⚠️ No datasets available. Please upload datasets first.")
                        for i in range(n_vars):
                            initial_conditions.append(0.0)
                
                st.session_state.initial_conditions = initial_conditions
                
                # Display current initial conditions summary
                with st.expander("📋 Initial Conditions Summary"):
                    ic_summary = []
                    for i, ic in enumerate(initial_conditions):
                        if var_names and i < len(var_names):
                            ic_summary.append(f"{var_names[i]}(0) = {ic}")
                        else:
                            ic_summary.append(f"y[{i}](0) = {ic}")
                    st.markdown("**Current Initial Conditions:**")
                    for summary in ic_summary:
                        st.markdown(f"- {summary}")
                
                # Data mapping
                st.write("**Data Mapping:**")
                st.info("Map your datasets to corresponding state variables for fitting")
                
                dataset_mapping = {}
                for dataset_name in st.session_state.datasets.keys():
                    st.markdown(f"**Map '{dataset_name}' to:**")
                    
                    # Create options for mapping
                    mapping_options = []
                    mapping_values = []
                    
                    for i in range(n_vars):
                        if var_names and i < len(var_names):
                            option_label = f"{var_names[i]} (state variable {i})"
                            mapping_options.append(option_label)
                        else:
                            option_label = f"y[{i}] (state variable {i})"
                            mapping_options.append(option_label)
                        mapping_values.append(i)
                    
                    # Determine current selection based on existing mapping
                    current_selection = 0
                    if dataset_name in st.session_state.dataset_mapping:
                        current_var_idx = st.session_state.dataset_mapping[dataset_name]
                        if current_var_idx < len(mapping_values):
                            current_selection = current_var_idx
                    
                    selected_idx = st.selectbox(
                        f"Variable for {dataset_name}:",
                        range(len(mapping_options)),
                        format_func=lambda x: mapping_options[x],
                        index=current_selection,
                        key=f"dataset_map_{dataset_name}_{job_key}"
                    )
                    
                    dataset_mapping[dataset_name] = mapping_values[selected_idx]
                
                st.session_state.dataset_mapping = dataset_mapping
                
                # Display mapping summary
                with st.expander("📊 Data Mapping Summary"):
                    st.markdown("**Current Mappings:**")
                    for dataset, var_idx in dataset_mapping.items():
                        if var_names and var_idx < len(var_names):
                            var_name = var_names[var_idx]
                            st.markdown(f"- **{dataset}** → {var_name} (y[{var_idx}])")
                        else:
                            st.markdown(f"- **{dataset}** → y[{var_idx}]")
            
            else:
                st.warning("Please define your ODE system first to auto-detect state variables.")
        else:
            st.info("Please upload datasets first to configure initial conditions and data mapping.")

# Tab 3: Enhanced Model Fitting
with tab3:
    st.header("Advanced Parameter Fitting")
    
    if st.session_state.param_names and st.session_state.datasets:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Parameter Configuration")
            
            # Parameter bounds input method selection
            bounds_input_method = st.radio(
                "Parameter Bounds Input Method",
                ["Individual Inputs", "Code-based Definition"],
                help="Choose how to define parameter bounds"
            )
            
            bounds = {}
            initial_guesses = {}
            
            # Check if we loaded an example
            example_bounds = {}
            example_initial = {}
            if use_example and selected_example in ODE_EXAMPLES:
                example_bounds = ODE_EXAMPLES[selected_example].get('bounds', {})
                example_initial = ODE_EXAMPLES[selected_example].get('initial_guess', {})
            
            if bounds_input_method == "Code-based Definition":
                st.markdown("### 🔧 Code-based Parameter Bounds")
                st.info("""
                Define parameter bounds using Python dictionary syntax. This allows for:
                - Easy copy/paste from research papers or existing code
                - Batch parameter definition
                - Mathematical expressions for bounds
                - Comments and documentation
                - Load from previously saved bounds files
                """)
                
                # File upload for bounds configuration
                col_upload, col_template = st.columns([1, 1])
                
                with col_upload:
                    uploaded_bounds_file = st.file_uploader(
                        "Upload Bounds Configuration File",
                        type=['py', 'txt'],
                        help="Upload a Python file containing bounds and initial_guess dictionaries"
                    )
                    
                    if uploaded_bounds_file is not None:
                        try:
                            bounds_content = uploaded_bounds_file.read().decode('utf-8')
                            st.session_state.uploaded_bounds_code = bounds_content
                            st.success("✅ Bounds file loaded successfully!")
                        except Exception as e:
                            st.error(f"❌ Error reading bounds file: {str(e)}")
                
                with col_template:
                    if st.button("📋 Generate Template", key="generate_template"):
                        template_code = """# mODEl Parameter Bounds Configuration Template - Dobrovolny Lab TCU
# Save this as a .py file for easy reuse in mODEl
# Software: mODEl by Dobrovolny Laboratory, Texas Christian University

# Example bounds for common parameters
bounds = {
    # Infection/transmission rates
    'beta': (1e-6, 10.0),
    'gamma': (1e-6, 10.0),
    
    # Growth/decay rates
    'alpha': (1e-6, 100.0),
    'delta': (1e-6, 10.0),
    
    # Production/clearance rates
    'p': (0.1, 1000.0),
    'c': (1e-6, 10.0),
    
    # Other parameters
    'k': (1e-6, 10.0),
    'K': (1.0, 1000.0),
    'r': (1e-6, 10.0),
    'rho': (1e-6, 10.0),
}

# Initial parameter guesses
initial_guess = {
    'beta': 0.5,
    'gamma': 0.1,
    'alpha': 1.0,
    'delta': 0.5,
    'p': 10.0,
    'c': 1.0,
    'k': 1.0,
    'K': 100.0,
    'r': 1.0,
    'rho': 0.1,
}

# You can also use mathematical expressions:
# bounds = {
#     'beta': (1e-6, 2.0),
#     'gamma': (1e-6, 1.0),
#     'K': (10.0, 10**3),  # Using exponents
#     'rate': (1e-6, np.pi),  # Using numpy constants
# }
"""
                        st.download_button(
                            label="Download mODEl Template",
                            data=template_code,
                            file_name="mODEl_bounds_template.py",
                            mime="text/plain"
                        )
                
                # Default bounds code template
                default_bounds_code = ""
                if hasattr(st.session_state, 'uploaded_bounds_code'):
                    default_bounds_code = st.session_state.uploaded_bounds_code
                elif st.session_state.param_names:
                    default_bounds_code = "# Parameter bounds definition\nbounds = {\n"
                    for param in st.session_state.param_names:
                        if param in example_bounds:
                            lower, upper = example_bounds[param]
                            default_bounds_code += f"    '{param}': ({lower}, {upper}),  # {param} bounds\n"
                        else:
                            default_bounds_code += f"    '{param}': (1e-6, 10.0),  # {param} bounds\n"
                    default_bounds_code += "}\n\n# Initial guesses\ninitial_guess = {\n"
                    for param in st.session_state.param_names:
                        if param in example_initial:
                            guess = example_initial[param]
                            default_bounds_code += f"    '{param}': {guess},  # {param} initial guess\n"
                        else:
                            default_bounds_code += f"    '{param}': 1.0,  # {param} initial guess\n"
                    default_bounds_code += "}"
                
                bounds_code = st.text_area(
                    "Parameter Bounds Code",
                    value=default_bounds_code,
                    height=300,
                    help="Define bounds and initial guesses using Python dictionary syntax"
                )
                
                # Parse and validate bounds code
                if bounds_code.strip():
                    try:
                        # Create a safe execution environment
                        exec_globals = {
                            '__builtins__': {},
                            'abs': abs, 'min': min, 'max': max,
                            'round': round, 'int': int, 'float': float,
                            'pow': pow, 'exp': np.exp, 'log': np.log,
                            'sqrt': np.sqrt, 'pi': np.pi, 'e': np.e,
                            'np': np, 'numpy': np
                        }
                        exec_locals = {}
                        
                        # Execute the bounds code
                        exec(bounds_code, exec_globals, exec_locals)
                        
                        # Extract bounds and initial guesses
                        if 'bounds' in exec_locals:
                            parsed_bounds = exec_locals['bounds']
                            if isinstance(parsed_bounds, dict):
                                bounds = parsed_bounds
                                st.success(f"✅ Parsed bounds for {len(bounds)} parameters")
                                
                                # Validate bounds
                                for param, bound in bounds.items():
                                    if not isinstance(bound, (tuple, list)) or len(bound) != 2:
                                        st.error(f"❌ Invalid bound format for {param}: {bound}")
                                        st.info("Bounds should be (lower, upper) tuples")
                                        break
                                    elif bound[0] >= bound[1]:
                                        st.error(f"❌ Invalid bound range for {param}: lower >= upper")
                                        break
                            else:
                                st.error("❌ 'bounds' must be a dictionary")
                        
                        if 'initial_guess' in exec_locals:
                            parsed_initial = exec_locals['initial_guess']
                            if isinstance(parsed_initial, dict):
                                initial_guesses = parsed_initial
                                st.success(f"✅ Parsed initial guesses for {len(initial_guesses)} parameters")
                            else:
                                st.error("❌ 'initial_guess' must be a dictionary")
                        
                        # Store parsed bounds for export
                        st.session_state.bounds_code = bounds_code
                        st.session_state.parsed_bounds = bounds
                        st.session_state.parsed_initial_guesses = initial_guesses
                        
                        # Advanced validation and analysis
                        if bounds and initial_guesses:
                            validation_messages = []
                            
                            # Check if initial guesses are within bounds
                            for param in st.session_state.param_names:
                                if param in bounds and param in initial_guesses:
                                    lower, upper = bounds[param]
                                    guess = initial_guesses[param]
                                    if guess < lower or guess > upper:
                                        validation_messages.append(f"⚠️ Initial guess for {param} ({guess}) is outside bounds ({lower}, {upper})")
                                elif param in bounds:
                                    validation_messages.append(f"⚠️ No initial guess specified for {param}")
                                elif param in initial_guesses:
                                    validation_messages.append(f"⚠️ No bounds specified for {param}")
                            
                            # Check for very wide or very narrow bounds
                            for param, (lower, upper) in bounds.items():
                                if upper / lower > 1e6:
                                    validation_messages.append(f"⚠️ Very wide bounds for {param} (ratio: {upper/lower:.1e})")
                                elif upper / lower < 10:
                                    validation_messages.append(f"ℹ️ Narrow bounds for {param} (ratio: {upper/lower:.1f})")
                            
                            if validation_messages:
                                with st.expander("🔍 Bounds Validation Analysis"):
                                    for message in validation_messages:
                                        if message.startswith("⚠️"):
                                            st.warning(message)
                                        else:
                                            st.info(message)
                        
                    except Exception as e:
                        st.error(f"❌ Error parsing bounds code: {str(e)}")
                        st.info("Check your Python syntax and variable names")
                        
                        # Provide helpful debugging information
                        with st.expander("🔧 Debugging Help"):
                            st.markdown("""
                            **Common syntax errors:**
                            - Missing commas between dictionary items
                            - Unmatched parentheses or brackets
                            - Incorrect indentation
                            - Missing quotes around parameter names
                            
                            **Valid examples:**
                            ```python
                            bounds = {
                                'beta': (1e-6, 10.0),
                                'gamma': (0.01, 1.0),
                            }
                            
                            initial_guess = {
                                'beta': 0.5,
                                'gamma': 0.1,
                            }
                            ```
                            """)
                
                # Quick load examples
                with st.expander("🚀 Quick Load Examples"):
                    st.markdown("**Load predefined bounds configurations for common ODE systems:**")
                    
                    example_configs = {
                        "Viral Dynamics": {
                            "description": "Bounds for viral infection models",
                            "code": """# Viral dynamics parameter bounds
bounds = {
    'beta': (1e-6, 10.0),    # Infection rate
    'gamma': (1e-6, 10.0),   # Conversion rate
    'delta': (1e-6, 10.0),   # Death rate
    'p': (0.1, 1000.0),      # Production rate
    'c': (1e-6, 10.0),       # Clearance rate
    'alpha': (1e-6, 100.0),  # Decay rate
    'rho': (1e-6, 10.0),     # Reversion rate
}

initial_guess = {
    'beta': 0.5,
    'gamma': 0.1,
    'delta': 0.5,
    'p': 10.0,
    'c': 1.0,
    'alpha': 1.0,
    'rho': 0.1,
}"""
                        },
                        "Epidemiological": {
                            "description": "Bounds for SIR/SEIR models",
                            "code": """# Epidemiological model bounds
bounds = {
    'beta': (0.0, 2.0),      # Transmission rate
    'gamma': (0.0, 1.0),     # Recovery rate
    'alpha': (0.0, 1.0),     # Incubation rate
    'mu': (0.0, 0.1),        # Birth/death rate
    'nu': (0.0, 0.5),        # Vaccination rate
}

initial_guess = {
    'beta': 0.3,
    'gamma': 0.1,
    'alpha': 0.2,
    'mu': 0.01,
    'nu': 0.05,
}"""
                        },
                        "Chemical Kinetics": {
                            "description": "Bounds for reaction kinetics",
                            "code": """# Chemical kinetics bounds
bounds = {
    'k1': (1e-6, 100.0),     # First-order rate constant
    'k2': (1e-6, 100.0),     # Second-order rate constant
    'K': (1.0, 1000.0),      # Equilibrium constant
    'Km': (0.1, 100.0),      # Michaelis constant
    'Vmax': (0.1, 1000.0),   # Maximum velocity
    'kcat': (1e-3, 1000.0),  # Catalytic constant
}

initial_guess = {
    'k1': 1.0,
    'k2': 0.5,
    'K': 10.0,
    'Km': 5.0,
    'Vmax': 10.0,
    'kcat': 1.0,
}"""
                        },
                        "Population Dynamics": {
                            "description": "Bounds for population models",
                            "code": """# Population dynamics bounds
bounds = {
    'r': (0.0, 5.0),         # Growth rate
    'K': (1.0, 10000.0),     # Carrying capacity
    'a': (0.0, 10.0),        # Interaction coefficient
    'b': (0.0, 1.0),         # Conversion efficiency
    'c': (0.0, 5.0),         # Mortality rate
    'd': (0.0, 1.0),         # Mortality coefficient
}

initial_guess = {
    'r': 1.0,
    'K': 100.0,
    'a': 0.1,
    'b': 0.1,
    'd': 0.1,
    'c': 0.5,
}"""
                        }
                    }
                    
                    selected_config = st.selectbox(
                        "Choose a configuration:",
                        [""] + list(example_configs.keys()),
                        key="quick_config_select"
                    )
                    
                    if selected_config:
                        config = example_configs[selected_config]
                        st.info(f"**{selected_config}:** {config['description']}")
                        
                        if st.button(f"Load {selected_config} Configuration", key=f"load_{selected_config}"):
                            st.session_state.uploaded_bounds_code = config['code']
                            st.rerun()
                
                # Display parsed bounds summary
                if bounds:
                    with st.expander("📋 Parsed Bounds Summary"):
                        bounds_summary = pd.DataFrame([
                            {
                                'Parameter': param,
                                'Lower Bound': bounds.get(param, (0, 0))[0],
                                'Upper Bound': bounds.get(param, (0, 0))[1],
                                'Initial Guess': initial_guesses.get(param, 'Not specified')
                            }
                            for param in st.session_state.param_names
                        ])
                        st.dataframe(bounds_summary, use_container_width=True)
                        
                        # Export bounds code
                        if st.button("📥 Export Bounds Code", key="export_bounds"):
                            bounds_export = f"""# mODEl Parameter Bounds Configuration - Dobrovolny Lab TCU
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Software: mODEl by Dobrovolny Laboratory, Texas Christian University

{bounds_code}

# Usage Instructions for mODEl:
# 1. Copy this code block
# 2. Paste into the 'Code-based Definition' section in mODEl
# 3. Modify bounds as needed
# 4. Click 'Run Advanced Model Fitting'
"""
                            st.download_button(
                                label="Download mODEl Bounds Configuration",
                                data=bounds_export,
                                file_name=f"mODEl_parameter_bounds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                                mime="text/plain"
                            )
            else:
                # Original individual inputs method
                for param in st.session_state.param_names:
                    st.markdown(f"**{param}**")
                    cols = st.columns(3)
                    with cols[0]:
                        default_lower = example_bounds.get(param, (1e-6, 10.0))[0] if param in example_bounds else 1e-6
                        lower = st.number_input(f"Lower", value=default_lower, format="%.2e", key=f"lower_{param}")
                    with cols[1]:
                        default_upper = example_bounds.get(param, (1e-6, 10.0))[1] if param in example_bounds else 10.0
                        upper = st.number_input(f"Upper", value=default_upper, format="%.2e", key=f"upper_{param}")
                    with cols[2]:
                        default_initial = example_initial.get(param, 1.0)
                        initial = st.number_input(f"Initial", value=default_initial, format="%.2e", key=f"initial_{param}")
                    
                    bounds[param] = (lower, upper)
                    initial_guesses[param] = initial
                    
                    # Display current bounds
                    st.caption(f"Bounds: [{lower:.2e}, {upper:.2e}], Initial: {initial:.2e}")
                    st.markdown("---")
        
        with col2:
            st.subheader("Fitting Options")
            
            # Objective function weights
            st.write("**Dataset Weights:**")
            dataset_weights = {}
            for dataset_name in st.session_state.datasets.keys():
                weight = st.number_input(f"Weight for {dataset_name}", value=1.0, min_value=0.0, key=f"weight_{dataset_name}")
                dataset_weights[dataset_name] = weight
            
            # Additional fitting options
            use_log_transform = st.checkbox("Use Log Transform for Positive Data", value=False)
            normalize_by_initial = st.checkbox("Normalize by Initial Values", value=False)
            
            # Show parameter bounds summary
            if bounds:
                st.write("**Parameter Bounds Summary:**")
                bounds_summary = pd.DataFrame([
                    {
                        'Parameter': param,
                        'Lower Bound': bounds[param][0],
                        'Upper Bound': bounds[param][1],
                        'Initial Guess': initial_guesses[param]
                    }
                    for param in st.session_state.param_names
                ])
                st.dataframe(bounds_summary, use_container_width=True)
            
            # Run fitting button
            if st.button("🚀 Run Advanced Model Fitting", type="primary"):
                # Store Tab 3 specific settings temporarily
                tab3_dataset_weights = dataset_weights.copy()
                tab3_use_log_transform = use_log_transform
                tab3_normalize_by_initial = normalize_by_initial
                
                # Create bounds dictionary for the reusable function
                if bounds:
                    st.session_state.parsed_bounds = bounds
                    st.session_state.parsed_initial_guesses = initial_guesses
                
                # Run the model fitting using reusable function
                success = run_model_fitting()
                
                # Update results with Tab 3 specific settings if successful
                if success and st.session_state.fit_results:
                    st.session_state.fit_results['dataset_weights'] = tab3_dataset_weights
                    st.session_state.fit_results['fitting_options'].update({
                        'use_log_transform': tab3_use_log_transform,
                        'normalize_by_initial': tab3_normalize_by_initial
                    })
                    
                    # Show additional Tab 3 specific success information
                    if tab3_use_log_transform:
                        st.info("📊 **Log transformation** was applied to positive data")
                    if tab3_normalize_by_initial:
                        st.info("📊 **Normalization** by initial values was applied")
                    if any(w != 1.0 for w in tab3_dataset_weights.values()):
                        weights_info = ", ".join([f"{name}: {weight}" for name, weight in tab3_dataset_weights.items() if weight != 1.0])
                        st.info(f"⚖️ **Custom dataset weights:** {weights_info}")
    else:
        st.warning("Please upload datasets and define your ODE system first to use mODEl's fitting capabilities.")

# Tab 4: Enhanced Results
with tab4:
    st.header("mODEl Fitting Results & Analysis")
    
    if st.session_state.fit_results:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Fitted Parameters")
            
            # Create results dataframe
            results_data = []
            for param, value in st.session_state.fit_results['params'].items():
                results_data.append({
                    'Parameter': param,
                    'Value': value,
                    'Initial Guess': initial_guesses.get(param, 'N/A'),
                    'Lower Bound': bounds.get(param, (None, None))[0],
                    'Upper Bound': bounds.get(param, (None, None))[1]
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Optimization info
            st.metric("Total Cost", f"{st.session_state.fit_results['cost']:.6e}")
            st.metric("Success", "✅ Yes" if st.session_state.fit_results['success'] else "❌ No")
            
            # Export results
            if st.button("📥 Export Results Package"):
                # Create a comprehensive results package
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create zip file in memory
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Parameters file
                    params_str = "# mODEl Results - Dobrovolny Lab TCU\n"
                    params_str += f"# Generated: {timestamp}\n"
                    params_str += "# https://github.com/DobrovolnyLab\n\n"
                    params_str += "# Fitted Parameters\n"
                    for param, value in st.session_state.fit_results['params'].items():
                        params_str += f"{param}: {value:.6e}\n"
                    params_str += f"\n# Cost: {st.session_state.fit_results['cost']:.6e}\n"
                    
                    # Add bounds code if available
                    if hasattr(st.session_state, 'bounds_code') and st.session_state.bounds_code:
                        params_str += "\n# Parameter Bounds Code Used\n"
                        params_str += "# " + "="*50 + "\n"
                        bounds_lines = st.session_state.bounds_code.split('\n')
                        for line in bounds_lines:
                            params_str += f"# {line}\n"
                        params_str += "# " + "="*50 + "\n"
                    
                    zip_file.writestr("fitted_parameters.txt", params_str)
                    
                    # Results CSV
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    zip_file.writestr("results_summary.csv", csv_buffer.getvalue())
                    
                    # Dataset info
                    dataset_info = "# mODEl Dataset Information - Dobrovolny Lab TCU\n"
                    for name, data in st.session_state.datasets.items():
                        dataset_info += f"\n{name}:\n"
                        dataset_info += f"  - Rows: {len(data)}\n"
                        dataset_info += f"  - Time range: {data['time'].min():.2f} - {data['time'].max():.2f}\n"
                        dataset_info += f"  - Value range: {data['value'].min():.2f} - {data['value'].max():.2f}\n"
                    zip_file.writestr("dataset_info.txt", dataset_info)
                    
                    # ODE system and bounds configuration
                    ode_config = "# mODEl System Configuration - Dobrovolny Lab TCU\n"
                    ode_config += f"# Generated: {timestamp}\n"
                    ode_config += "# https://github.com/DobrovolnyLab\n\n"
                    ode_config += "# ODE System Definition\n"
                    ode_config += f"# Number of state variables: {len(st.session_state.initial_conditions)}\n"
                    ode_config += f"# Initial conditions: {st.session_state.initial_conditions}\n\n"
                    ode_config += "# ODE Code:\n"
                    ode_lines = st.session_state.ode_system.split('\n')
                    for line in ode_lines:
                        ode_config += f"# {line}\n"
                    
                    # Add bounds code if available
                    if hasattr(st.session_state, 'bounds_code') and st.session_state.bounds_code:
                        ode_config += "\n# Parameter Bounds Code:\n"
                        ode_config += st.session_state.bounds_code
                    
                    zip_file.writestr("ode_configuration.py", ode_config)
                    
                    # Analysis summary
                    analysis_summary = f"""# mODEl Analysis Summary - Dobrovolny Lab TCU
# Generated: {timestamp}
# Software: mODEl by Dobrovolny Laboratory, Texas Christian University

## Model Information
- Number of parameters: {len(st.session_state.param_names)}
- Parameters: {', '.join(st.session_state.param_names)}
- Number of datasets: {len(st.session_state.datasets)}
- Datasets: {', '.join(st.session_state.datasets.keys())}

## Optimization Results
- Final cost: {st.session_state.fit_results['cost']:.6e}
- Optimization successful: {st.session_state.fit_results['success']}
- Optimization method: {st.session_state.fit_results.get('method', 'Not specified')}

## Fitted Parameters
"""
                    for param, value in st.session_state.fit_results['params'].items():
                        analysis_summary += f"- {param}: {value:.6e}\n"
                    
                    if hasattr(st.session_state, 'bootstrap_results') and st.session_state.bootstrap_results:
                        analysis_summary += "\n## Bootstrap Analysis\n"
                        analysis_summary += f"- Bootstrap samples: {st.session_state.bootstrap_results['n_samples']}\n"
                        analysis_summary += f"- Confidence level: {st.session_state.bootstrap_results['confidence_level']}%\n"
                        analysis_summary += f"- Bootstrap method: {st.session_state.bootstrap_results['method']}\n"
                    
                    analysis_summary += "\n---\nGenerated by mODEl - Dobrovolny Laboratory, Texas Christian University"
                    
                    zip_file.writestr("analysis_summary.md", analysis_summary)
                
                zip_buffer.seek(0)
                st.download_button(
                    label="Download mODEl Results Package",
                    data=zip_buffer.getvalue(),
                    file_name=f"mODEl_results_{timestamp}.zip",
                    mime="application/zip"
                )
        
        with col2:
            st.subheader("Model Fit Visualization")
            
            try:
                # Validate ODE system before visualization
                def create_ode_func(param_names, ode_code):
                    # Properly indent the user's ODE code
                    lines = ode_code.strip().split('\n')
                    indented_lines = []
                    for line in lines:
                        if line.strip():  # Only indent non-empty lines
                            indented_lines.append('    ' + line.strip())
                        else:
                            indented_lines.append('')
                    
                    indented_code = '\n'.join(indented_lines)
                    
                    func_code = f"""
def ode_system(y, t, {', '.join(param_names)}):
{indented_code}
"""
                    exec(func_code, globals())
                    return globals()['ode_system']
                
                ode_func = create_ode_func(st.session_state.param_names, st.session_state.ode_system)
                
                # Validate ODE system compatibility
                fitted_params = [st.session_state.fit_results['params'][p] for p in st.session_state.param_names]
                test_result = ode_func(st.session_state.initial_conditions, 0, *fitted_params)
                
                validation_passed = True
                if len(test_result) != len(st.session_state.initial_conditions):
                    st.error(f"❌ ODE system mismatch: Your ODE returns {len(test_result)} derivatives but you have {len(st.session_state.initial_conditions)} initial conditions.")
                    st.error("Cannot generate visualization. Please fix your ODE definition or initial conditions.")
                    validation_passed = False
                
                if validation_passed:
                    # Generate model predictions
                    all_times = []
                    for data in st.session_state.datasets.values():
                        all_times.extend(data['time'].values)
                    
                    t_min, t_max = min(all_times), max(all_times)
                    t_fine = np.linspace(t_min, t_max, 1000)
                    
                    # Solve with fitted parameters
                    solution = odeint(ode_func, st.session_state.initial_conditions, t_fine, 
                                    args=tuple(fitted_params))
                    
                    # Create visualization
                    if st.session_state.visualization_settings['plot_style'] == "plotly":
                        fig = go.Figure()
                        
                        # Add experimental data
                        for dataset_name, data in st.session_state.datasets.items():
                            fig.add_trace(go.Scatter(
                                x=data['time'],
                                y=data['value'],
                                mode='markers',
                                name=f'Data: {dataset_name}',
                                marker=dict(size=8)
                            ))
                        
                        # Add model predictions
                        for dataset_name, data in st.session_state.datasets.items():
                            var_idx = st.session_state.dataset_mapping[dataset_name]
                            fig.add_trace(go.Scatter(
                                x=t_fine,
                                y=solution[:, var_idx],
                                mode='lines',
                                name=f'Model: {dataset_name}',
                                line=dict(width=3)
                            ))
                        
                        fig.update_layout(
                            title="Multi-Dataset Model Fit",
                            xaxis_title="Time",
                            yaxis_title="Value",
                            hovermode='x unified',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        # Matplotlib plot
                        n_datasets = len(st.session_state.datasets)
                        fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 5))
                        if n_datasets == 1:
                            axes = [axes]
                        
                        for i, (dataset_name, data) in enumerate(st.session_state.datasets.items()):
                            var_idx = st.session_state.dataset_mapping[dataset_name]
                            
                            axes[i].scatter(data['time'], data['value'], 
                                          label=f'Data: {dataset_name}', alpha=0.7, s=50)
                            axes[i].plot(t_fine, solution[:, var_idx], 
                                       label=f'Model: {dataset_name}', linewidth=2)
                            axes[i].set_xlabel('Time')
                            axes[i].set_ylabel('Value')
                            axes[i].set_title(f'{dataset_name}')
                            axes[i].legend()
                            axes[i].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Model statistics
                    with st.expander("📊 Detailed Model Statistics"):
                        st.write("**Dataset-wise Statistics:**")
                        for dataset_name, data in st.session_state.datasets.items():
                            var_idx = st.session_state.dataset_mapping[dataset_name]
                            model_vals = np.interp(data['time'], t_fine, solution[:, var_idx])
                            
                            residuals = data['value'] - model_vals
                            rmse = np.sqrt(np.mean(residuals**2))
                            mae = np.mean(np.abs(residuals))
                            
                            if np.std(data['value']) > 0:
                                r_squared = 1 - (np.sum(residuals**2) / np.sum((data['value'] - np.mean(data['value']))**2))
                            else:
                                r_squared = np.nan
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric(f"RMSE ({dataset_name})", f"{rmse:.4f}")
                            col2.metric(f"MAE ({dataset_name})", f"{mae:.4f}")
                            col3.metric(f"R² ({dataset_name})", f"{r_squared:.4f}")
                
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
                st.exception(e)
    
    else:
        st.info("No fitting results available. Please run the model fitting first.")

# Tab 5: Bootstrap Analysis
with tab5:
    st.header("mODEl Bootstrap Analysis for Parameter Uncertainty")
    
    if not st.session_state.fit_results:
        st.warning("Please run mODEl model fitting first before bootstrap analysis.")
    else:
        st.markdown("""
        <div class="bootstrap-warning">
        <strong>⚠️ Bootstrap Analysis</strong><br>
        This analysis provides uncertainty estimates for fitted parameters. 
        It can be computationally intensive for large datasets or complex models.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Bootstrap Configuration")
            
            n_bootstrap_samples = st.number_input("Number of Bootstrap Samples", 
                                                value=st.session_state.bootstrap_settings['n_samples'], min_value=10, max_value=1000)
            
            bootstrap_method = st.selectbox("Bootstrap Method", 
                                          ["Residual Resampling", "Parametric Bootstrap"],
                                          index=["Residual Resampling", "Parametric Bootstrap"].index(st.session_state.bootstrap_settings['method']))
            
            confidence_level = st.selectbox("Confidence Level", [90, 95, 99], 
                                           index=[90, 95, 99].index(st.session_state.bootstrap_settings['confidence_level']))
            
            # Update session state with current selections
            st.session_state.bootstrap_settings['n_samples'] = n_bootstrap_samples
            st.session_state.bootstrap_settings['method'] = bootstrap_method
            st.session_state.bootstrap_settings['confidence_level'] = confidence_level
            
            # Run Bootstrap Analysis Button
            if st.button("🎯 Run Bootstrap Analysis", type="primary"):
                with st.spinner("Running bootstrap analysis..."):
                    try:
                        # Create ODE function for bootstrap
                        def create_ode_func(param_names, ode_code):
                            # Properly indent the user's ODE code
                            lines = ode_code.strip().split('\n')
                            indented_lines = []
                            for line in lines:
                                if line.strip():  # Only indent non-empty lines
                                    indented_lines.append('    ' + line.strip())
                                else:
                                    indented_lines.append('')
                            
                            indented_code = '\n'.join(indented_lines)
                            
                            func_code = f"""
def ode_system(y, t, {', '.join(param_names)}):
{indented_code}
"""
                            exec(func_code, globals())
                            return globals()['ode_system']
                        
                        ode_func = create_ode_func(st.session_state.param_names, st.session_state.ode_system)
                        
                        # Get fitted parameters
                        fitted_params = [st.session_state.fit_results['params'][p] for p in st.session_state.param_names]
                        
                        # Prepare datasets for bootstrap
                        all_times = []
                        for data in st.session_state.datasets.values():
                            all_times.extend(data['time'].values)
                        unique_times = sorted(set(all_times))
                        t_data = np.array(unique_times)
                        
                        # Calculate residuals for the original fit
                        sol_orig = odeint(ode_func, st.session_state.initial_conditions, t_data, 
                                        args=tuple(fitted_params))
                        
                        original_residuals = {}
                        for dataset_name, data in st.session_state.datasets.items():
                            var_idx = st.session_state.dataset_mapping[dataset_name]
                            model_vals = np.interp(data['time'], t_data, sol_orig[:, var_idx])
                            original_residuals[dataset_name] = data['value'] - model_vals
                        
                        # Bootstrap analysis
                        bootstrap_params = []
                        
                        # Initialize progress tracking
                        if 'bootstrap_logs' not in st.session_state:
                            st.session_state.bootstrap_logs = []
                        st.session_state.bootstrap_logs = []  # Clear previous logs
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(n_bootstrap_samples):
                            try:
                            # Create bootstrap datasets
                            bootstrap_datasets = {}
                            
                                if bootstrap_method == "Residual Resampling":
                                    # Resample residuals
                                    for dataset_name, data in st.session_state.datasets.items():
                                    var_idx = st.session_state.dataset_mapping[dataset_name]
                                        model_vals = np.interp(data['time'], t_data, sol_orig[:, var_idx])
                                        
                                        # Resample residuals
                                        resampled_residuals = np.random.choice(
                                            original_residuals[dataset_name], 
                                            size=len(original_residuals[dataset_name]), 
                                            replace=True
                                        )
                                        
                                        # Create new bootstrap data
                                        bootstrap_values = model_vals + resampled_residuals
                                bootstrap_datasets[dataset_name] = pd.DataFrame({
                                    'time': data['time'],
                                            'value': bootstrap_values
                                        })
                                
                                else:  # Parametric Bootstrap
                                    # Add noise based on residual variance
                                    for dataset_name, data in st.session_state.datasets.items():
                                        var_idx = st.session_state.dataset_mapping[dataset_name]
                                        model_vals = np.interp(data['time'], t_data, sol_orig[:, var_idx])
                                        
                                        # Calculate residual standard deviation
                                        residual_std = np.std(original_residuals[dataset_name])
                                        
                                        # Add random noise
                                        noise = np.random.normal(0, residual_std, len(model_vals))
                                        bootstrap_values = model_vals + noise
                                        
                                        bootstrap_datasets[dataset_name] = pd.DataFrame({
                                            'time': data['time'],
                                            'value': bootstrap_values
                                        })
                                
                                # Fit to bootstrap data
                                def bootstrap_objective(params):
                                    try:
                                        sol = odeint(ode_func, st.session_state.initial_conditions, t_data, 
                                                   args=tuple(params))
                                        
                                        total_ssr = 0
                                        for dataset_name, data in bootstrap_datasets.items():
                                            var_idx = st.session_state.dataset_mapping[dataset_name]
                                            model_vals = np.interp(data['time'], t_data, sol[:, var_idx])
                                            ssr = np.sum((model_vals - data['value'])**2)
                                            total_ssr += ssr
                                        
                                        return total_ssr
                                    except:
                                        return 1e12
                                
                                # Use original fitted parameters as starting point
                                bounds = []
                                for param in st.session_state.param_names:
                                    if hasattr(st.session_state, 'parsed_bounds') and param in st.session_state.parsed_bounds:
                                        bounds.append(st.session_state.parsed_bounds[param])
                                    else:
                                        # Default bounds around fitted value
                                        fitted_val = st.session_state.fit_results['params'][param]
                                        bounds.append((fitted_val * 0.01, fitted_val * 100))
                                
                                # Optimize bootstrap sample
                                result = minimize(bootstrap_objective, fitted_params, 
                                                method=st.session_state.optimization_settings['method'], 
                                                bounds=bounds)
                                
                                if result.success:
                            bootstrap_params.append(result.x)
                                    
                                    # Log progress every 10 samples
                                    if (i + 1) % 10 == 0 or i == 0:
                                        param_summary = {}
                                        for j, param in enumerate(st.session_state.param_names):
                                            values = [bp[j] for bp in bootstrap_params]
                                            param_summary[param] = {
                                                'mean': np.mean(values),
                                                'std': np.std(values),
                                                'current': result.x[j]
                                            }
                                        
                                        log_entry = f"Sample {i+1}/{n_bootstrap_samples}: "
                                        log_entry += ", ".join([f"{p}={param_summary[p]['current']:.3e}" 
                                                              for p in st.session_state.param_names[:3]])
                                        if len(st.session_state.param_names) > 3:
                                            log_entry += "..."
                                        
                                        st.session_state.bootstrap_logs.append(log_entry)
                            
                            except Exception as e:
                                # Skip failed samples
                                pass
                            
                            # Update progress
                            progress = (i + 1) / n_bootstrap_samples
                            progress_bar.progress(progress)
                            status_text.text(f"Bootstrap sample {i+1}/{n_bootstrap_samples} "
                                           f"({len(bootstrap_params)} successful)")
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        if len(bootstrap_params) > 0:
                            # Calculate statistics
                            bootstrap_stats = {}
                            for i, param in enumerate(st.session_state.param_names):
                                values = [bp[i] for bp in bootstrap_params]
                                
                                # Calculate confidence intervals
                        alpha = (100 - confidence_level) / 100
                                ci_lower = np.percentile(values, 100 * alpha / 2)
                                ci_upper = np.percentile(values, 100 * (1 - alpha / 2))
                        
                            bootstrap_stats[param] = {
                                    'values': values,
                                    'mean': np.mean(values),
                                    'std': np.std(values),
                                    'ci_lower': ci_lower,
                                    'ci_upper': ci_upper
                                }
                            
                            # Store results
                        st.session_state.bootstrap_results = {
                            'n_samples': n_bootstrap_samples,
                                'method': bootstrap_method,
                            'confidence_level': confidence_level,
                                'successful_samples': len(bootstrap_params),
                                'stats': bootstrap_stats
                            }
                            
                            st.success(f"✅ Bootstrap analysis completed! "
                                     f"({len(bootstrap_params)}/{n_bootstrap_samples} successful samples)")
                        
                        else:
                            st.error("❌ Bootstrap analysis failed - no successful samples")
                        
                    except Exception as e:
                        st.error(f"❌ Bootstrap analysis error: {str(e)}")
                        st.exception(e)
        
        with col2:
            if not hasattr(st.session_state, 'bootstrap_logs') or not st.session_state.bootstrap_logs:
                st.subheader("Bootstrap Progress & Logs")
                st.info("👆 Configure bootstrap settings and click 'Run Bootstrap Analysis' to see live progress here.")
                st.markdown("""
                **Features:**
                - 🔄 Real-time progress updates
                - 📋 Live parameter logs
                - 📊 Running statistics
                - ⚙️ Configurable log frequency
                - 🎯 Adjustable display limits
                """)
            
            # Show final bootstrap results if available
            if st.session_state.bootstrap_results:
                st.markdown("---")
                st.subheader("Final Bootstrap Results")
                
                # Summary statistics
                stats_data = []
                for param, stats in st.session_state.bootstrap_results['stats'].items():
                    stats_data.append({
                        'Parameter': param,
                        'Original': st.session_state.fit_results['params'][param],
                        'Bootstrap Mean': stats['mean'],
                        'Bootstrap Std': stats['std'],
                        f'{confidence_level}% CI Lower': stats['ci_lower'],
                        f'{confidence_level}% CI Upper': stats['ci_upper']
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
                
                # Export bootstrap results
                if st.button("📥 Export Bootstrap Results"):
                    bootstrap_export = f"""# mODEl Bootstrap Analysis Results - Dobrovolny Lab TCU
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Software: mODEl by Dobrovolny Laboratory, Texas Christian University

## Configuration
- Bootstrap samples: {st.session_state.bootstrap_results['n_samples']}
- Confidence level: {st.session_state.bootstrap_results['confidence_level']}%
- Method: {st.session_state.bootstrap_results['method']}

## Parameter Statistics
"""
                    for param, stats in st.session_state.bootstrap_results['stats'].items():
                        bootstrap_export += f"""
### {param}
- Original estimate: {st.session_state.fit_results['params'][param]:.6e}
- Bootstrap mean: {stats['mean']:.6e}
- Bootstrap std: {stats['std']:.6e}
- {st.session_state.bootstrap_results['confidence_level']}% CI: [{stats['ci_lower']:.6e}, {stats['ci_upper']:.6e}]
"""
                    
                    bootstrap_export += "\n---\nGenerated by mODEl - Dobrovolny Laboratory, Texas Christian University"
                    
                    st.download_button(
                        label="Download mODEl Bootstrap Analysis",
                        data=bootstrap_export,
                        file_name=f"mODEl_bootstrap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/plain"
                    )

        # Parameter distribution plots
        if st.session_state.bootstrap_results and st.session_state.visualization_settings['show_distributions']:
            st.subheader("Parameter Distribution Analysis")
            
            n_params = len(st.session_state.param_names)
            confidence_level = st.session_state.bootstrap_results['confidence_level']
            
            if st.session_state.visualization_settings['plot_style'] == "plotly":
                # Use Plotly for parameter distribution plots
                cols = 3
                rows = (n_params + cols - 1) // cols
                
                fig = make_subplots(
                    rows=rows, cols=cols,
                    subplot_titles=[param for param in st.session_state.param_names],
                    vertical_spacing=0.1,
                    horizontal_spacing=0.1
                )
                
                for i, param in enumerate(st.session_state.param_names):
                    row = (i // cols) + 1
                    col = (i % cols) + 1
                    
                    values = st.session_state.bootstrap_results['stats'][param]['values']
                    
                    # Add histogram
                    fig.add_trace(
                        go.Histogram(
                            x=values,
                            name=f'{param} Distribution',
                            marker_color='skyblue',
                            opacity=0.7,
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                    
                    # Add vertical lines for statistics
                    y_max = np.histogram(values, bins=20)[0].max()
                    
                    # Original estimate
                    fig.add_trace(
                        go.Scatter(
                            x=[st.session_state.fit_results['params'][param], 
                               st.session_state.fit_results['params'][param]],
                            y=[0, y_max],
                            mode='lines',
                            line=dict(color='red', dash='dash', width=2),
                            name='Original' if i == 0 else None,
                            showlegend=True if i == 0 else False,
                            legendgroup='original'
                        ),
                        row=row, col=col
                    )
                    
                    # Bootstrap mean
                    fig.add_trace(
                        go.Scatter(
                            x=[st.session_state.bootstrap_results['stats'][param]['mean'], 
                               st.session_state.bootstrap_results['stats'][param]['mean']],
                            y=[0, y_max],
                            mode='lines',
                            line=dict(color='green', width=2),
                            name='Bootstrap Mean' if i == 0 else None,
                            showlegend=True if i == 0 else False,
                            legendgroup='bootstrap_mean'
                        ),
                        row=row, col=col
                    )
                    
                    # Confidence interval bounds
                    fig.add_trace(
                        go.Scatter(
                            x=[st.session_state.bootstrap_results['stats'][param]['ci_lower'], 
                               st.session_state.bootstrap_results['stats'][param]['ci_lower']],
                            y=[0, y_max],
                            mode='lines',
                            line=dict(color='orange', dash='dot', width=2),
                            name=f'{confidence_level}% CI' if i == 0 else None,
                            showlegend=True if i == 0 else False,
                            legendgroup='ci'
                        ),
                        row=row, col=col
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[st.session_state.bootstrap_results['stats'][param]['ci_upper'], 
                               st.session_state.bootstrap_results['stats'][param]['ci_upper']],
                            y=[0, y_max],
                            mode='lines',
                            line=dict(color='orange', dash='dot', width=2),
                            name=None,
                            showlegend=False,
                            legendgroup='ci'
                        ),
                        row=row, col=col
                    )
                
                fig.update_layout(
                    height=400*rows,
                    title_text="Parameter Distribution Analysis",
                    template='plotly_white'
                )
                fig.update_xaxes(title_text="Parameter Value")
                fig.update_yaxes(title_text="Frequency")
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Use matplotlib for parameter distribution plots
                cols = 3
                rows = (n_params + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
                if rows == 1:
                    axes = axes.reshape(1, -1)
                if cols == 1:
                    axes = axes.reshape(-1, 1)
                
                for i, param in enumerate(st.session_state.param_names):
                    row, col = i // cols, i % cols
                    
                    if rows == 1:
                        ax = axes[col] if cols > 1 else axes[0]
                    else:
                        ax = axes[row, col] if cols > 1 else axes[row]
                    
                    values = st.session_state.bootstrap_results['stats'][param]['values']
                    ax.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    
                    # Add vertical lines for statistics
                    ax.axvline(st.session_state.fit_results['params'][param], 
                              color='red', linestyle='--', linewidth=2, label='Original')
                    ax.axvline(st.session_state.bootstrap_results['stats'][param]['mean'], 
                              color='green', linestyle='-', linewidth=2, label='Bootstrap Mean')
                    ax.axvline(st.session_state.bootstrap_results['stats'][param]['ci_lower'], 
                              color='orange', linestyle=':', linewidth=2, label=f'{confidence_level}% CI')
                    ax.axvline(st.session_state.bootstrap_results['stats'][param]['ci_upper'], 
                              color='orange', linestyle=':', linewidth=2)
                    
                    ax.set_title(f'{param}')
                    ax.set_xlabel('Parameter Value')
                    ax.set_ylabel('Frequency')
                    if i == 0:
                        ax.legend()
                
                # Remove empty subplots
                for i in range(n_params, rows * cols):
                    row, col = i // cols, i % cols
                    fig.delaxes(axes[row, col] if cols > 1 else axes[row])
                
                plt.tight_layout()
                st.pyplot(fig)

# Tab 6: Examples (same as before)
with tab6:
    st.header("📚 ODE System Examples")
    st.markdown("Explore common ODE systems and their applications.")
    
    for name, example in ODE_EXAMPLES.items():
        with st.expander(f"**{name}**"):
            st.markdown(f"**Description:** {example['description']}")
            st.code(example['code'], language='python')
            st.markdown(f"**Parameters:** {', '.join(example['parameters'])}")
            
            # Show typical parameter ranges
            if 'bounds' in example:
                bounds_str = "\n".join([f"- {p}: [{example['bounds'][p][0]}, {example['bounds'][p][1]}]" 
                                      for p in example['parameters']])
                st.markdown(f"**Typical Parameter Ranges:**\n{bounds_str}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>mODEl: Advanced ODE Model Fitting</strong></p>
    <p><em>Developed by Arjan Suri and Sahaj Satani</em></p>
    <p><em>Dobrovolny Laboratory | Texas Christian University</em></p>
    <p style='font-size: 0.8em; color: #666;'>
        For support and documentation, visit: 
        <a href='https://personal.tcu.edu/hdobrovolny' target='_blank'>personal.tcu.edu/hdobrovolny</a>
    </p>
</div>
""", unsafe_allow_html=True) 