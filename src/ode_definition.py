"""
mODEl ODE Definition Module
Handles ODE system definition, parameter detection, initial conditions, and data mapping
"""

import streamlit as st
import numpy as np
import pandas as pd
import re
from typing import List, Tuple, Dict
from ode_examples import ODE_EXAMPLES


def detect_state_variables(ode_code: str) -> Tuple[int, List[str]]:
    """Detect the number of state variables from ODE code"""
    lines = ode_code.strip().split('\n')
    max_y_index = -1
    unpacked_vars = []
    
    for line in lines:
        line = line.strip()
        
        # Check for variable unpacking (e.g., "T, R, I, V, F = y")
        if '= y' in line and not line.startswith('#'):
            var_part = line.split('= y')[0].strip()
            var_part = var_part.split('#')[0].strip()
            if ',' in var_part:
                unpacked_vars = [v.strip() for v in var_part.split(',')]
                return len(unpacked_vars), unpacked_vars
        
        # Check for y[i] indexing
        y_indices = re.findall(r'y\[(\d+)\]', line)
        for idx_str in y_indices:
            idx = int(idx_str)
            max_y_index = max(max_y_index, idx)
    
    if max_y_index >= 0:
        return max_y_index + 1, []
    
    return 1, []


def extract_parameter_names(ode_code: str, var_names: List[str]) -> List[str]:
    """Extract parameter names from ODE code"""
    param_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
    all_names = re.findall(param_pattern, ode_code)
    exclude = {'y', 'dydt', 'return', 'def', 'if', 'else', 'for', 'while', 'in', 'and', 'or', 'not', 't', 'N', 
              'dTdt', 'dRdt', 'dIdt', 'dVdt', 'dFdt', 'T', 'R', 'I', 'V', 'F', 'dxdt', 'dx', 'dt'}
    exclude.update(var_names)
    param_names = list(set(all_names) - exclude)
    return sorted(param_names)


def render_ode_definition_tab():
    """Render the ODE Definition tab content"""
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
            n_vars_detected, var_names = detect_state_variables(ode_code)
            
            # Update session state with detected variables
            st.session_state.auto_detected_vars = n_vars_detected
            st.session_state.auto_detected_var_names = var_names
            
            # Extract parameter names
            param_names = extract_parameter_names(ode_code, var_names)
            
            if param_names:
                st.session_state.param_names = param_names
                
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
                
                st.write("**Initial Conditions:**")
                st.markdown(f"*Setting initial conditions for {n_vars} state variables*")
                
                # Simple manual input for initial conditions
                initial_conditions = []
                job_key = st.session_state.active_job if st.session_state.active_job else 'main'
                
                if n_vars <= 3:
                    cols = st.columns(n_vars)
                    for i in range(n_vars):
                        with cols[i]:
                            if var_names and i < len(var_names):
                                label = f"{var_names[i]}(0)"
                            else:
                                label = f"y[{i}](0)"
                            
                            current_value = 0.0
                            if i < len(st.session_state.initial_conditions):
                                current_value = st.session_state.initial_conditions[i]
                            
                            ic = st.number_input(
                                label, 
                                value=current_value, 
                                key=f"manual_ic_{i}_{job_key}"
                            )
                            initial_conditions.append(ic)
                else:
                    for i in range(n_vars):
                        if var_names and i < len(var_names):
                            label = f"{var_names[i]}(0)"
                        else:
                            label = f"y[{i}](0)"
                        
                        current_value = 0.0
                        if i < len(st.session_state.initial_conditions):
                            current_value = st.session_state.initial_conditions[i]
                        
                        ic = st.number_input(
                            label, 
                            value=current_value, 
                            key=f"manual_ic_{i}_{job_key}"
                        )
                        initial_conditions.append(ic)
                
                st.session_state.initial_conditions = initial_conditions
                
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
                            st.markdown(f"- **{dataset}** → y[{var_idx}])")
            
            else:
                st.warning("Please define your ODE system first to auto-detect state variables.")
        else:
            st.info("Please upload datasets first to configure initial conditions and data mapping.") 