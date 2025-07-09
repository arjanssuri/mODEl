"""
mODEl Model Fitting Module
Contains all model fitting functionality, ODE processing, and optimization routines
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import re
import hashlib


@st.cache_resource(ttl=1800)  # Cache for 30 minutes - use cache_resource for functions
def create_ode_function_cached(param_names: List[str], ode_code: str):
    """Create an ODE function from code string with caching"""
    # Create a unique key for this function based on parameters and code
    func_key = f"{param_names}_{ode_code}"
    
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
    
    # Create a local namespace for the function
    local_namespace = {}
    exec(func_code, globals(), local_namespace)
    return local_namespace['ode_system']


def create_ode_function(param_names: List[str], ode_code: str):
    """Create an ODE function from code string (non-cached version for immediate use)"""
    return create_ode_function_cached(param_names, ode_code)


@st.cache_data
def solve_ode_system_cached(ode_code: str, param_names: List[str], initial_conditions: List[float], 
                           t_data: np.ndarray, parameters: List[float]) -> np.ndarray:
    """Cache ODE solutions for identical parameter sets"""
    try:
        # Create function for this specific solve (not cached separately)
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
        
        # Create a local namespace for the function
        local_namespace = {}
        exec(func_code, globals(), local_namespace)
        ode_func = local_namespace['ode_system']
        
        # Solve the ODE
        solution = odeint(ode_func, initial_conditions, t_data, args=tuple(parameters))
        return solution
    except Exception as e:
        # Return empty array if solve fails
        return np.array([])


def solve_ode_system(ode_code: str, param_names: List[str], initial_conditions: List[float], 
                    t_data: np.ndarray, parameters: List[float]) -> np.ndarray:
    """Solve ODE system with caching"""
    return solve_ode_system_cached(ode_code, param_names, initial_conditions, t_data, parameters)


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
    
    # Filter out common Python keywords, variables, and detected variable names
    exclude = {'y', 'dydt', 'return', 'def', 'if', 'else', 'for', 'while', 'in', 'and', 'or', 'not', 't', 'N', 
              'dTdt', 'dRdt', 'dIdt', 'dVdt', 'dFdt', 'T', 'R', 'I', 'V', 'F', 'dxdt', 'dx', 'dt'}
    exclude.update(var_names)
    
    param_names = list(set(all_names) - exclude)
    return sorted(param_names)


def validate_model_setup() -> Tuple[bool, List[str]]:
    """Validate that model is ready for fitting"""
    missing_items = []
    
    if not st.session_state.param_names:
        missing_items.append("Define ODE system and parameters")
    
    if not st.session_state.datasets:
        missing_items.append("Upload datasets")
    
    if not st.session_state.initial_conditions:
        missing_items.append("Set initial conditions")
    
    if not st.session_state.dataset_mapping:
        missing_items.append("Map datasets to state variables")
    
    return len(missing_items) == 0, missing_items


def create_model_hash(ode_system: str, param_names: List[str], initial_conditions: List[float],
                     datasets: Dict, dataset_mapping: Dict, bounds: Dict, settings: Dict) -> str:
    """Create a hash of the complete model setup for caching results"""
    try:
        # Combine all model components
        model_str = f"{ode_system}_{param_names}_{initial_conditions}_{dataset_mapping}_{bounds}_{settings}"
        
        # Add dataset info
        for name, data in datasets.items():
            model_str += f"{name}_{len(data)}_{data['time'].sum()}_{data['value'].sum()}"
        
        return hashlib.md5(model_str.encode()).hexdigest()
    except:
        return ""


def run_model_fitting() -> bool:
    """Run model fitting with current settings - callable from anywhere"""
    from utils import invalidate_cache_if_needed
    
    # Check for cache invalidation
    invalidate_cache_if_needed()
    
    # Validation checks
    is_ready, missing_items = validate_model_setup()
    if not is_ready:
        for item in missing_items:
            st.error(f"❌ {item}")
        return False
    
    # Create model hash for result caching
    model_hash = create_model_hash(
        st.session_state.ode_system,
        st.session_state.param_names,
        st.session_state.initial_conditions,
        st.session_state.datasets,
        st.session_state.dataset_mapping,
        getattr(st.session_state, 'parsed_bounds', {}),
        st.session_state.optimization_settings
    )
    
    # Check if we have cached results for this exact model
    if (hasattr(st.session_state, 'cached_fit_results') and 
        st.session_state.get('cached_model_hash') == model_hash and
        st.session_state.cached_fit_results is not None):
        
        st.session_state.fit_results = st.session_state.cached_fit_results
        st.info("🚀 Using cached model fitting results!")
        return True
    
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
            # Create ODE function for testing
            ode_func = create_ode_function(st.session_state.param_names, st.session_state.ode_system)
            
            # Test ODE function
            test_params = [initial_guesses[param] for param in st.session_state.param_names]
            test_result = ode_func(st.session_state.initial_conditions, 0, *test_params)
            
            if len(test_result) != len(st.session_state.initial_conditions):
                st.error(f"❌ ODE system mismatch: Your ODE returns {len(test_result)} derivatives but you have {len(st.session_state.initial_conditions)} initial conditions.")
                return False
            
            # Prepare all datasets
            all_times = []
            for data in st.session_state.datasets.values():
                all_times.extend(data['time'].values)
            
            # Get unique sorted time points
            unique_times = sorted(set(all_times))
            t_data = np.array(unique_times)
            
            # Multi-objective optimization function
            def objective(params):
                try:
                    # Use cached ODE solving
                    sol = solve_ode_system(
                        st.session_state.ode_system,
                        st.session_state.param_names,
                        st.session_state.initial_conditions,
                        t_data,
                        params
                    )
                    
                    if sol.size == 0:  # Failed solve
                        return 1e12
                    
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
            fit_results = {
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
            
            # Cache the results
            st.session_state.fit_results = fit_results
            st.session_state.cached_fit_results = fit_results
            st.session_state.cached_model_hash = model_hash
            
            # Success message with parameter summary
            param_summary = ", ".join([f"{param}={value:.3e}" for param, value in fit_results['params'].items()])
            st.success(f"✅ mODEl fitting completed! Cost: {result.fun:.3e}")
            st.info(f"📊 **Fitted Parameters:** {param_summary}")
            
            return True
    
    except Exception as e:
        st.error(f"❌ Model fitting error: {str(e)}")
        return False 