"""
mODEl Results Analysis Module
Handles results display, visualization, and export functionality
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.integrate import odeint
from datetime import datetime
import io
import zipfile
from model_fitting import create_ode_function


def render_results_analysis_tab():
    """Render the Results Analysis tab content"""
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
                    'Initial Guess': st.session_state.parsed_initial_guesses.get(param, 'N/A') if hasattr(st.session_state, 'parsed_initial_guesses') else 'N/A',
                    'Lower Bound': st.session_state.parsed_bounds.get(param, (None, None))[0] if hasattr(st.session_state, 'parsed_bounds') else 'N/A',
                    'Upper Bound': st.session_state.parsed_bounds.get(param, (None, None))[1] if hasattr(st.session_state, 'parsed_bounds') else 'N/A'
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
                # Create ODE function
                ode_func = create_ode_function(st.session_state.param_names, st.session_state.ode_system)
                
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