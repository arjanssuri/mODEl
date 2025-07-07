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

# Set page configuration
st.set_page_config(
    page_title="ODE Model Fitting Tool",
    page_icon="📊",
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
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .parameter-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .bootstrap-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🧮 Advanced ODE Model Fitting Tool")
st.markdown("""
This tool provides comprehensive ODE modeling capabilities including:
- Multi-dataset upload and fitting
- Bootstrap analysis for parameter uncertainty
- Advanced visualization and result export
- Support for complex multi-variable systems
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

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Optimization method selection
    opt_method = st.selectbox(
        "Optimization Method",
        ["L-BFGS-B", "Nelder-Mead", "SLSQP", "Powell", "TNC", "Differential Evolution"],
        help="Select the optimization algorithm for parameter fitting"
    )
    
    # Tolerance settings
    st.subheader("Convergence Settings")
    tol = st.number_input("Tolerance", value=1e-8, format="%.2e", help="Convergence tolerance")
    max_iter = st.number_input("Max Iterations", value=1000, min_value=100, step=100)
    
    # Advanced options
    st.subheader("Advanced Options")
    use_relative_error = st.checkbox("Use Relative Error", value=True, help="Use relative error instead of absolute error")
    multi_start = st.checkbox("Multi-start Optimization", value=False)
    if multi_start:
        n_starts = st.number_input("Number of starts", value=10, min_value=2, max_value=100)
    
    # Bootstrap settings
    st.subheader("Bootstrap Analysis")
    enable_bootstrap = st.checkbox("Enable Bootstrap Analysis", value=False)
    if enable_bootstrap:
        n_bootstrap = st.number_input("Bootstrap Samples", value=100, min_value=10, max_value=1000)
        bootstrap_method = st.selectbox("Bootstrap Method", ["Residual Resampling", "Parametric Bootstrap"])
    
    # Plot settings
    st.subheader("Visualization Settings")
    plot_style = st.selectbox("Plot Style", ["seaborn", "plotly"])
    show_phase_portrait = st.checkbox("Show Phase Portrait (2D systems)", value=False)
    show_distributions = st.checkbox("Show Parameter Distributions", value=False)

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📁 Data Upload", 
    "🧬 ODE Definition", 
    "📊 Model Fitting", 
    "📈 Results", 
    "🎯 Bootstrap Analysis",
    "📚 Examples"
])

# Tab 1: Enhanced Data Upload
with tab1:
    st.header("Upload Experimental Data")
    
    # Multi-dataset upload
    st.subheader("Multi-Dataset Upload")
    st.info("Upload multiple datasets for different variables in your ODE system. Each dataset should have 'time' and 'value' columns.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dynamic dataset upload
        dataset_name = st.text_input("Dataset Name", placeholder="e.g., viral_load, interferon")
        uploaded_file = st.file_uploader(
            "Choose a file (txt or csv)",
            type=['txt', 'csv'],
            help="Upload experimental data with 'time' and 'value' columns"
        )
        
        if uploaded_file is not None and dataset_name:
            try:
                # Read the file
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_csv(uploaded_file, delimiter='\t')
                
                # Validate columns
                if 'time' not in data.columns or 'value' not in data.columns:
                    st.error("Data must have 'time' and 'value' columns")
                else:
                    st.session_state.datasets[dataset_name] = data
                    st.success(f"✅ Dataset '{dataset_name}' loaded successfully!")
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
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
    
    # Data visualization
    if st.session_state.datasets:
        st.subheader("Data Visualization")
        
        # Create combined plot
        fig = plt.figure(figsize=(12, 8))
        
        n_datasets = len(st.session_state.datasets)
        if n_datasets == 1:
            # Single plot
            for name, data in st.session_state.datasets.items():
                plt.plot(data['time'], data['value'], 'o-', label=name)
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.title(f'Dataset: {name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
        else:
            # Subplots for multiple datasets
            cols = 2
            rows = (n_datasets + cols - 1) // cols
            
            for i, (name, data) in enumerate(st.session_state.datasets.items()):
                plt.subplot(rows, cols, i + 1)
                plt.plot(data['time'], data['value'], 'o-', label=name)
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.title(f'Dataset: {name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Data statistics
        with st.expander("📊 Data Statistics"):
            for name, data in st.session_state.datasets.items():
                st.write(f"**{name}**")
                st.write(data.describe())
                st.write("---")

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
        T, R, I, V, F = y  # Unpack variables
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
            
            # Extract parameter names
            param_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            all_names = re.findall(param_pattern, ode_code)
            # Filter out common Python keywords and variables
            exclude = {'y', 'dydt', 'return', 'def', 'if', 'else', 'for', 'while', 'in', 'and', 'or', 'not', 't', 'N', 
                      'dTdt', 'dRdt', 'dIdt', 'dVdt', 'dFdt', 'T', 'R', 'I', 'V', 'F', 'dxdt', 'dx', 'dt'}
            param_names = list(set(all_names) - exclude)
            
            if param_names:
                st.session_state.param_names = sorted(param_names)
                st.success(f"Detected parameters: {', '.join(st.session_state.param_names)}")
    
    with col2:
        st.subheader("Initial Conditions & Data Mapping")
        
        if st.session_state.datasets:
            # Get number of variables from ODE system
            if st.session_state.ode_system:
                # Try to determine number of variables
                n_vars = st.number_input("Number of state variables", value=1, min_value=1, max_value=10)
                
                st.write("**Initial Conditions:**")
                initial_conditions = []
                for i in range(n_vars):
                    ic = st.number_input(f"y[{i}](0)", value=0.0, key=f"ic_{i}")
                    initial_conditions.append(ic)
                
                st.session_state.initial_conditions = initial_conditions
                
                # Data mapping
                st.write("**Data Mapping:**")
                st.info("Map your datasets to corresponding state variables for fitting")
                
                dataset_mapping = {}
                for dataset_name in st.session_state.datasets.keys():
                    var_index = st.selectbox(
                        f"Map '{dataset_name}' to variable:",
                        range(n_vars),
                        format_func=lambda x: f"y[{x}]",
                        key=f"map_{dataset_name}"
                    )
                    dataset_mapping[dataset_name] = var_index
                
                st.session_state.dataset_mapping = dataset_mapping

# Tab 3: Enhanced Model Fitting
with tab3:
    st.header("Advanced Parameter Fitting")
    
    if st.session_state.param_names and st.session_state.datasets:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Parameter Configuration")
            
            bounds = {}
            initial_guesses = {}
            
            # Check if we loaded an example
            example_bounds = {}
            example_initial = {}
            if use_example and selected_example in ODE_EXAMPLES:
                example_bounds = ODE_EXAMPLES[selected_example].get('bounds', {})
                example_initial = ODE_EXAMPLES[selected_example].get('initial_guess', {})
            
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
            
            # Run fitting button
            if st.button("🚀 Run Advanced Model Fitting", type="primary"):
                with st.spinner("Running optimization..."):
                    try:
                        # Create ODE function
                        def create_ode_func(param_names, ode_code):
                            func_code = f"""
def ode_system(y, t, {', '.join(param_names)}):
    {ode_code}
"""
                            exec(func_code, globals())
                            return globals()['ode_system']
                        
                        ode_func = create_ode_func(st.session_state.param_names, st.session_state.ode_system)
                        
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
                                    
                                    # Apply transformations
                                    if use_log_transform and np.all(data['value'] > 0):
                                        model_vals = np.log(np.maximum(model_vals, 1e-10))
                                        data_vals = np.log(data['value'])
                                    else:
                                        data_vals = data['value']
                                    
                                    # Calculate error
                                    if use_relative_error:
                                        error = ((model_vals - data_vals) / (np.abs(data_vals) + 1e-10))**2
                                    else:
                                        error = (model_vals - data_vals)**2
                                    
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
                        if multi_start:
                            best_result = None
                            best_cost = np.inf
                            
                            progress_bar = st.progress(0)
                            for i in range(n_starts):
                                # Random initial point
                                x0_random = []
                                for (low, high) in opt_bounds:
                                    if np.isfinite(low) and np.isfinite(high):
                                        x0_random.append(np.random.uniform(low, high))
                                    else:
                                        x0_random.append(np.random.lognormal(0, 1))
                                
                                result = minimize(objective, x0_random, method=opt_method, 
                                                bounds=opt_bounds, options={'maxiter': max_iter})
                                
                                if result.fun < best_cost:
                                    best_result = result
                                    best_cost = result.fun
                                
                                progress_bar.progress((i + 1) / n_starts)
                            
                            result = best_result
                            progress_bar.empty()
                        else:
                            result = minimize(objective, x0, method=opt_method, 
                                            bounds=opt_bounds, options={'maxiter': max_iter})
                        
                        # Store results
                        st.session_state.fit_results = {
                            'params': dict(zip(st.session_state.param_names, result.x)),
                            'cost': result.fun,
                            'success': result.success,
                            'message': result.message,
                            'result_obj': result,
                            'dataset_weights': dataset_weights,
                            'fitting_options': {
                                'use_relative_error': use_relative_error,
                                'use_log_transform': use_log_transform,
                                'normalize_by_initial': normalize_by_initial
                            }
                        }
                        
                        st.success("✅ Model fitting completed!")
                        
                    except Exception as e:
                        st.error(f"Error during fitting: {str(e)}")
                        st.exception(e)
    else:
        st.warning("Please upload datasets and define your ODE system first.")

# Tab 4: Enhanced Results
with tab4:
    st.header("Fitting Results & Analysis")
    
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
                    params_str = "# ODE Model Fitting Results\n"
                    params_str += f"# Timestamp: {timestamp}\n\n"
                    params_str += "# Fitted Parameters\n"
                    for param, value in st.session_state.fit_results['params'].items():
                        params_str += f"{param}: {value:.6e}\n"
                    params_str += f"\n# Cost: {st.session_state.fit_results['cost']:.6e}\n"
                    zip_file.writestr("fitted_parameters.txt", params_str)
                    
                    # Results CSV
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    zip_file.writestr("results_summary.csv", csv_buffer.getvalue())
                    
                    # Dataset info
                    dataset_info = "# Dataset Information\n"
                    for name, data in st.session_state.datasets.items():
                        dataset_info += f"\n{name}:\n"
                        dataset_info += f"  - Rows: {len(data)}\n"
                        dataset_info += f"  - Time range: {data['time'].min():.2f} - {data['time'].max():.2f}\n"
                        dataset_info += f"  - Value range: {data['value'].min():.2f} - {data['value'].max():.2f}\n"
                    zip_file.writestr("dataset_info.txt", dataset_info)
                
                zip_buffer.seek(0)
                st.download_button(
                    label="Download Results Package",
                    data=zip_buffer.getvalue(),
                    file_name=f"ode_results_{timestamp}.zip",
                    mime="application/zip"
                )
        
        with col2:
            st.subheader("Model Fit Visualization")
            
            try:
                # Generate model predictions
                all_times = []
                for data in st.session_state.datasets.values():
                    all_times.extend(data['time'].values)
                
                t_min, t_max = min(all_times), max(all_times)
                t_fine = np.linspace(t_min, t_max, 1000)
                
                # Create ODE function
                def create_ode_func(param_names, ode_code):
                    func_code = f"""
def ode_system(y, t, {', '.join(param_names)}):
    {ode_code}
"""
                    exec(func_code, globals())
                    return globals()['ode_system']
                
                ode_func = create_ode_func(st.session_state.param_names, st.session_state.ode_system)
                
                # Solve with fitted parameters
                fitted_params = [st.session_state.fit_results['params'][p] for p in st.session_state.param_names]
                solution = odeint(ode_func, st.session_state.initial_conditions, t_fine, 
                                args=tuple(fitted_params))
                
                # Create visualization
                if plot_style == "plotly":
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
    st.header("Bootstrap Analysis for Parameter Uncertainty")
    
    if not st.session_state.fit_results:
        st.warning("Please run model fitting first before bootstrap analysis.")
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
                                                value=100, min_value=10, max_value=1000)
            
            bootstrap_method = st.selectbox("Bootstrap Method", 
                                          ["Residual Resampling", "Parametric Bootstrap"])
            
            confidence_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)
            
            if st.button("🎯 Run Bootstrap Analysis", type="primary"):
                with st.spinner(f"Running bootstrap analysis with {n_bootstrap_samples} samples..."):
                    try:
                        # Create ODE function
                        def create_ode_func(param_names, ode_code):
                            func_code = f"""
def ode_system(y, t, {', '.join(param_names)}):
    {ode_code}
"""
                            exec(func_code, globals())
                            return globals()['ode_system']
                        
                        ode_func = create_ode_func(st.session_state.param_names, st.session_state.ode_system)
                        
                        # Get original fitted parameters
                        best_params = [st.session_state.fit_results['params'][p] for p in st.session_state.param_names]
                        
                        # Calculate residuals for each dataset
                        all_residuals = {}
                        all_times = []
                        for data in st.session_state.datasets.values():
                            all_times.extend(data['time'].values)
                        
                        t_min, t_max = min(all_times), max(all_times)
                        t_fine = np.linspace(t_min, t_max, 1000)
                        
                        # Solve with best parameters
                        solution = odeint(ode_func, st.session_state.initial_conditions, t_fine, 
                                        args=tuple(best_params))
                        
                        # Calculate residuals for each dataset
                        for dataset_name, data in st.session_state.datasets.items():
                            var_idx = st.session_state.dataset_mapping[dataset_name]
                            model_vals = np.interp(data['time'], t_fine, solution[:, var_idx])
                            residuals = data['value'] - model_vals
                            all_residuals[dataset_name] = residuals
                        
                        # Bootstrap optimization function
                        def bootstrap_objective(params, bootstrap_datasets):
                            try:
                                all_times_boot = []
                                for data in bootstrap_datasets.values():
                                    all_times_boot.extend(data['time'].values)
                                
                                t_boot = np.linspace(min(all_times_boot), max(all_times_boot), 500)
                                sol = odeint(ode_func, st.session_state.initial_conditions, t_boot, 
                                           args=tuple(params))
                                
                                total_ssr = 0
                                for dataset_name, data in bootstrap_datasets.items():
                                    var_idx = st.session_state.dataset_mapping[dataset_name]
                                    model_vals = np.interp(data['time'], t_boot, sol[:, var_idx])
                                    
                                    if st.session_state.fit_results['fitting_options']['use_relative_error']:
                                        error = ((model_vals - data['value']) / (np.abs(data['value']) + 1e-10))**2
                                    else:
                                        error = (model_vals - data['value'])**2
                                    
                                    total_ssr += np.sum(error)
                                
                                return total_ssr
                            except:
                                return 1e12
                        
                        # Run bootstrap
                        bootstrap_params = []
                        progress_bar = st.progress(0)
                        
                        for i in range(n_bootstrap_samples):
                            # Create bootstrap datasets
                            bootstrap_datasets = {}
                            
                            for dataset_name, data in st.session_state.datasets.items():
                                if bootstrap_method == "Residual Resampling":
                                    # Resample residuals
                                    boot_residuals = np.random.choice(all_residuals[dataset_name], 
                                                                     size=len(all_residuals[dataset_name]), 
                                                                     replace=True)
                                    var_idx = st.session_state.dataset_mapping[dataset_name]
                                    model_vals = np.interp(data['time'], t_fine, solution[:, var_idx])
                                    boot_values = model_vals + boot_residuals
                                else:
                                    # Parametric bootstrap (add noise based on residual variance)
                                    noise_std = np.std(all_residuals[dataset_name])
                                    boot_values = data['value'] + np.random.normal(0, noise_std, len(data))
                                
                                bootstrap_datasets[dataset_name] = pd.DataFrame({
                                    'time': data['time'],
                                    'value': boot_values
                                })
                            
                            # Optimize with bootstrap data
                            opt_bounds = [(bounds[param][0], bounds[param][1]) for param in st.session_state.param_names]
                            
                            result = minimize(bootstrap_objective, best_params, 
                                            args=(bootstrap_datasets,), 
                                            method='Nelder-Mead', bounds=opt_bounds)
                            
                            bootstrap_params.append(result.x)
                            progress_bar.progress((i + 1) / n_bootstrap_samples)
                        
                        progress_bar.empty()
                        
                        # Calculate statistics
                        bootstrap_params = np.array(bootstrap_params)
                        
                        alpha = (100 - confidence_level) / 100
                        ci_lower = (alpha / 2) * 100
                        ci_upper = (100 - alpha / 2) * 100
                        
                        bootstrap_stats = {}
                        for i, param in enumerate(st.session_state.param_names):
                            param_values = bootstrap_params[:, i]
                            bootstrap_stats[param] = {
                                'mean': np.mean(param_values),
                                'std': np.std(param_values),
                                'median': np.median(param_values),
                                'ci_lower': np.percentile(param_values, ci_lower),
                                'ci_upper': np.percentile(param_values, ci_upper),
                                'values': param_values
                            }
                        
                        st.session_state.bootstrap_results = {
                            'params': bootstrap_params,
                            'stats': bootstrap_stats,
                            'n_samples': n_bootstrap_samples,
                            'confidence_level': confidence_level,
                            'method': bootstrap_method
                        }
                        
                        st.success("✅ Bootstrap analysis completed!")
                        
                    except Exception as e:
                        st.error(f"Error during bootstrap analysis: {str(e)}")
                        st.exception(e)
        
        with col2:
            if st.session_state.bootstrap_results:
                st.subheader("Bootstrap Results")
                
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
        
        # Parameter distribution plots
        if st.session_state.bootstrap_results and show_distributions:
            st.subheader("Parameter Distribution Analysis")
            
            n_params = len(st.session_state.param_names)
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
    <p>Advanced ODE Model Fitting Tool | Built with Streamlit 🚀</p>
    <p><em>Supports multi-dataset fitting, bootstrap analysis, and comprehensive uncertainty quantification</em></p>
</div>
""", unsafe_allow_html=True) 