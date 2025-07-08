"""
mODEl Parameter Fitting Module
Handles advanced parameter configuration, bounds setting, and model fitting execution
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from model_fitting import run_model_fitting
from ode_examples import ODE_EXAMPLES


def render_parameter_fitting_tab():
    """Render the Parameter Fitting tab content"""
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
            
            if bounds_input_method == "Code-based Definition":
                st.markdown("### 🔧 Code-based Parameter Bounds")
                st.info("""
                Define parameter bounds using Python dictionary syntax. This allows for:
                - Easy copy/paste from research papers or existing code
                - Batch parameter definition
                - Mathematical expressions for bounds
                - Comments and documentation
                """)
                
                # Default bounds code template
                default_bounds_code = ""
                if st.session_state.param_names:
                    default_bounds_code = "# Parameter bounds definition\nbounds = {\n"
                    for param in st.session_state.param_names:
                        default_bounds_code += f"    '{param}': (1e-6, 10.0),  # {param} bounds\n"
                    default_bounds_code += "}\n\n# Initial guesses\ninitial_guess = {\n"
                    for param in st.session_state.param_names:
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
                        
                    except Exception as e:
                        st.error(f"❌ Error parsing bounds code: {str(e)}")
                
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
                        if st.button("📥 Export Bounds Code"):
                            bounds_export = f"""# mODEl Parameter Bounds Configuration - Dobrovolny Lab TCU
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

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
                        lower = st.number_input(f"Lower", value=1e-6, format="%.2e", key=f"lower_{param}")
                    with cols[1]:
                        upper = st.number_input(f"Upper", value=10.0, format="%.2e", key=f"upper_{param}")
                    with cols[2]:
                        initial = st.number_input(f"Initial", value=1.0, format="%.2e", key=f"initial_{param}")
                    
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