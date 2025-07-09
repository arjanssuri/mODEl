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


def auto_detect_initial_condition_method():
    """Auto-detect the best initial condition method based on dataset mapping"""
    if (hasattr(st.session_state, 'dataset_mapping') and 
        st.session_state.dataset_mapping and 
        len(st.session_state.dataset_mapping) > 0):
        
        # If we have dataset mapping, switch to dataset-specific first values
        if 'ic_method' not in st.session_state:
            st.session_state.ic_method = "Use Dataset-Specific First Values"
        
        return "Use Dataset-Specific First Values"
    
    # Default to manual input if no mapping
    return st.session_state.get('ic_method', "Manual Input")


def create_auto_mapping_from_dataset_mapping():
    """Create automatic mapping selections based on current dataset mapping"""
    auto_selections = {}
    
    if hasattr(st.session_state, 'dataset_mapping') and st.session_state.dataset_mapping:
        # For each state variable, find which dataset maps to it
        for dataset_name, var_idx in st.session_state.dataset_mapping.items():
            auto_selections[var_idx] = dataset_name
    
    return auto_selections


def render_ode_definition_tab():
    """Render the ODE Definition tab content"""
    st.header("Define Your ODE System")
    
    # Prevent form submission on Enter key
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && event.target.type !== 'submit') {
            event.preventDefault();
            return false;
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Add example selector
    use_example = st.checkbox("Use an example ODE system", key="use_example_ode")
    
    if use_example:
        selected_example = st.selectbox(
            "Select an ODE system:",
            list(ODE_EXAMPLES.keys()),
            key="example_ode_selector"
        )
        
        if selected_example:
            example = ODE_EXAMPLES[selected_example]
            st.info(f"**{selected_example}**: {example['description']}")
            
            if st.button("Load Example", key="load_example_ode"):
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
            placeholder="# Define your ODE system here\n# Example:\n# dxdt = -k * x\n# return [dxdt]",
            key="ode_system_input"
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
                
                # Option to override auto-detection
                with st.expander("🔧 Override Auto-Detection"):
                    st.markdown("**Auto-detection found:** " + 
                              (f"{n_vars} variables ({', '.join(var_names)})" if var_names else f"{n_vars} variables (y[0] to y[{n_vars-1}])"))
                    
                    override_detection = st.checkbox("Override auto-detection", value=False, key="override_auto_detection")
                    if override_detection:
                        n_vars = st.number_input("Manual number of state variables", value=n_vars, min_value=1, max_value=10, key="manual_n_vars")
                        st.info(f"Using manual setting: {n_vars} variables")
                
                st.write("**Data Mapping:**")
                st.info("Map your datasets to corresponding state variables for fitting")
                
                dataset_mapping = {}
                dataset_mapping_changed = False
                old_mapping = st.session_state.dataset_mapping.copy() if st.session_state.dataset_mapping else {}
                
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
                        key=f"dataset_map_{dataset_name}"
                    )
                    
                    dataset_mapping[dataset_name] = mapping_values[selected_idx]
                    
                    # Check if mapping changed
                    if old_mapping.get(dataset_name) != mapping_values[selected_idx]:
                        dataset_mapping_changed = True
                
                st.session_state.dataset_mapping = dataset_mapping
                
                # Auto-detect initial condition method if mapping changed
                if dataset_mapping_changed and dataset_mapping:
                    st.session_state.ic_method = "Use Dataset-Specific First Values"
                    # Create auto-selections based on mapping
                    st.session_state.auto_ic_selections = create_auto_mapping_from_dataset_mapping()
                    st.info("🎯 **Auto-switched to dataset-specific initial values based on your mapping!**")
                
                # Display mapping summary
                with st.expander("📊 Data Mapping Summary"):
                    st.markdown("**Current Mappings:**")
                    for dataset, var_idx in dataset_mapping.items():
                        if var_names and var_idx < len(var_names):
                            var_name = var_names[var_idx]
                            st.markdown(f"- **{dataset}** → {var_name} (y[{var_idx}])")
                        else:
                            st.markdown(f"- **{dataset}** → y[{var_idx}]")
                
                st.write("**Initial Conditions:**")
                st.markdown(f"*Setting initial conditions for {n_vars} state variables*")
                
                # Option to set initial conditions from first data values
                with st.expander("🔧 Initial Condition Options"):
                    # Auto-detect the best method
                    auto_detected_method = auto_detect_initial_condition_method()
                    
                    ic_method = st.radio(
                        "How to set initial conditions:",
                        ["Manual Input", "Use First Data Values", "Use Dataset-Specific First Values"],
                        index=["Manual Input", "Use First Data Values", "Use Dataset-Specific First Values"].index(auto_detected_method),
                        help="Choose how to set initial conditions for state variables",
                        key="ic_method_radio"
                    )
                    
                    # Store the method
                    st.session_state.ic_method = ic_method
                    
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
                        - 🎯 **Automatically selected based on your data mapping!**
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
                        
                        # Get auto-selections if they exist
                        auto_selections = getattr(st.session_state, 'auto_ic_selections', {})
                        
                        for i in range(n_vars):
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                if var_names and i < len(var_names):
                                    var_label = f"{var_names[i]}(0)"
                                else:
                                    var_label = f"y[{i}](0)"
                                
                                # Determine default selection based on auto-detection
                                default_index = 0  # "Manual"
                                if i in auto_selections and auto_selections[i] in dataset_names:
                                    default_index = dataset_names.index(auto_selections[i]) + 1  # +1 because "Manual" is first
                                
                                # Dropdown to select dataset with stable key
                                selected_dataset = st.selectbox(
                                    f"Dataset for {var_label}:",
                                    ["Manual"] + dataset_names,
                                    index=default_index,
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
            
            else:
                st.warning("Please define your ODE system first to auto-detect state variables.")
        else:
            st.info("Please upload datasets first to configure initial conditions and data mapping.") 