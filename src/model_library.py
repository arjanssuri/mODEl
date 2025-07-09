"""
mODEl Model Library Module
Handles saving, loading, and managing ODE models and parameter bounds
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import io


def initialize_model_library():
    """Initialize the model library in session state"""
    if 'saved_models' not in st.session_state:
        st.session_state.saved_models = {}
    if 'saved_bounds' not in st.session_state:
        st.session_state.saved_bounds = {}


def save_current_model():
    """Save the current ODE model to the library"""
    if not st.session_state.ode_system or not st.session_state.param_names:
        st.error("❌ No ODE system defined to save!")
        return False
    
    # Get model name from user
    model_name = st.text_input(
        "Model Name:",
        value="",
        placeholder="e.g., Viral_Dynamics_Model",
        key="save_model_name",
        help="Enter a name for this ODE model"
    )
    
    if not model_name:
        st.warning("Please enter a model name")
        return False
    
    # Create model data
    model_data = {
        'name': model_name,
        'ode_system': st.session_state.ode_system,
        'param_names': st.session_state.param_names,
        'initial_conditions': st.session_state.initial_conditions,
        'auto_detected_vars': st.session_state.auto_detected_vars,
        'auto_detected_var_names': st.session_state.auto_detected_var_names,
        'dataset_mapping': st.session_state.dataset_mapping,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': st.text_area(
            "Description (optional):",
            placeholder="Brief description of this model...",
            key="save_model_description",
            height=100
        )
    }
    
    if st.button("💾 Save Model", type="primary", key="confirm_save_model"):
        st.session_state.saved_models[model_name] = model_data
        st.success(f"✅ Model '{model_name}' saved successfully!")
        return True
    
    return False


def save_current_bounds():
    """Save the current parameter bounds to the library"""
    if not hasattr(st.session_state, 'parsed_bounds') or not st.session_state.parsed_bounds:
        st.error("❌ No parameter bounds defined to save!")
        return False
    
    # Get bounds name from user
    bounds_name = st.text_input(
        "Bounds Configuration Name:",
        value="",
        placeholder="e.g., Standard_Viral_Bounds",
        key="save_bounds_name",
        help="Enter a name for this bounds configuration"
    )
    
    if not bounds_name:
        st.warning("Please enter a bounds configuration name")
        return False
    
    # Create bounds data
    bounds_data = {
        'name': bounds_name,
        'bounds_code': st.session_state.bounds_code,
        'parsed_bounds': st.session_state.parsed_bounds,
        'parsed_initial_guesses': st.session_state.parsed_initial_guesses,
        'param_names': st.session_state.param_names,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': st.text_area(
            "Description (optional):",
            placeholder="Brief description of these parameter bounds...",
            key="save_bounds_description",
            height=100
        )
    }
    
    if st.button("💾 Save Bounds", type="primary", key="confirm_save_bounds"):
        st.session_state.saved_bounds[bounds_name] = bounds_data
        st.success(f"✅ Bounds '{bounds_name}' saved successfully!")
        return True
    
    return False


def load_saved_model(model_name: str):
    """Load a saved model into the current session"""
    if model_name not in st.session_state.saved_models:
        st.error(f"❌ Model '{model_name}' not found!")
        return False
    
    model_data = st.session_state.saved_models[model_name]
    
    # Load model data into session state
    st.session_state.ode_system = model_data['ode_system']
    st.session_state.param_names = model_data['param_names']
    st.session_state.initial_conditions = model_data.get('initial_conditions', [])
    st.session_state.auto_detected_vars = model_data.get('auto_detected_vars', 1)
    st.session_state.auto_detected_var_names = model_data.get('auto_detected_var_names', [])
    st.session_state.dataset_mapping = model_data.get('dataset_mapping', {})
    
    st.success(f"✅ Model '{model_name}' loaded successfully!")
    return True


def load_saved_bounds(bounds_name: str):
    """Load saved parameter bounds into the current session"""
    if bounds_name not in st.session_state.saved_bounds:
        st.error(f"❌ Bounds '{bounds_name}' not found!")
        return False
    
    bounds_data = st.session_state.saved_bounds[bounds_name]
    
    # Load bounds data into session state
    st.session_state.bounds_code = bounds_data['bounds_code']
    st.session_state.parsed_bounds = bounds_data['parsed_bounds']
    st.session_state.parsed_initial_guesses = bounds_data['parsed_initial_guesses']
    
    st.success(f"✅ Bounds '{bounds_name}' loaded successfully!")
    return True


def export_model_library():
    """Export all saved models and bounds as JSON"""
    export_data = {
        'models': st.session_state.saved_models,
        'bounds': st.session_state.saved_bounds,
        'exported': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': '1.0'
    }
    
    json_str = json.dumps(export_data, indent=2)
    return json_str


def import_model_library(json_data: str):
    """Import models and bounds from JSON data"""
    try:
        data = json.loads(json_data)
        
        # Validate format
        if 'models' not in data or 'bounds' not in data:
            st.error("❌ Invalid model library format!")
            return False
        
        # Import models
        imported_models = 0
        for model_name, model_data in data['models'].items():
            if model_name not in st.session_state.saved_models:
                st.session_state.saved_models[model_name] = model_data
                imported_models += 1
        
        # Import bounds
        imported_bounds = 0
        for bounds_name, bounds_data in data['bounds'].items():
            if bounds_name not in st.session_state.saved_bounds:
                st.session_state.saved_bounds[bounds_name] = bounds_data
                imported_bounds += 1
        
        st.success(f"✅ Imported {imported_models} models and {imported_bounds} bounds configurations!")
        return True
        
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON format!")
        return False


def render_model_library_ui():
    """Render the model library UI"""
    st.subheader("📚 Model Library")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💾 Save", "📂 Load", "📊 Manage", "🔄 Import/Export"])
    
    with tab1:
        st.markdown("### Save Current Work")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Save ODE Model**")
            if st.session_state.ode_system and st.session_state.param_names:
                save_current_model()
            else:
                st.info("Define an ODE system first to save it")
        
        with col2:
            st.markdown("**Save Parameter Bounds**")
            if hasattr(st.session_state, 'parsed_bounds') and st.session_state.parsed_bounds:
                save_current_bounds()
            else:
                st.info("Define parameter bounds first to save them")
    
    with tab2:
        st.markdown("### Load Saved Work")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Load ODE Model**")
            if st.session_state.saved_models:
                model_names = list(st.session_state.saved_models.keys())
                selected_model = st.selectbox(
                    "Select model:",
                    [""] + model_names,
                    key="load_model_select"
                )
                
                if selected_model:
                    model_info = st.session_state.saved_models[selected_model]
                    st.info(f"**Created:** {model_info['created']}")
                    if model_info.get('description'):
                        st.info(f"**Description:** {model_info['description']}")
                    st.info(f"**Parameters:** {', '.join(model_info['param_names'])}")
                    
                    if st.button("🔄 Load Model", key="load_model_btn"):
                        load_saved_model(selected_model)
                        st.rerun()
            else:
                st.info("No saved models available")
        
        with col2:
            st.markdown("**Load Parameter Bounds**")
            if st.session_state.saved_bounds:
                bounds_names = list(st.session_state.saved_bounds.keys())
                selected_bounds = st.selectbox(
                    "Select bounds:",
                    [""] + bounds_names,
                    key="load_bounds_select"
                )
                
                if selected_bounds:
                    bounds_info = st.session_state.saved_bounds[selected_bounds]
                    st.info(f"**Created:** {bounds_info['created']}")
                    if bounds_info.get('description'):
                        st.info(f"**Description:** {bounds_info['description']}")
                    st.info(f"**Parameters:** {', '.join(bounds_info['param_names'])}")
                    
                    if st.button("🔄 Load Bounds", key="load_bounds_btn"):
                        load_saved_bounds(selected_bounds)
                        st.rerun()
            else:
                st.info("No saved bounds available")
    
    with tab3:
        st.markdown("### Manage Library")
        
        # Display saved models
        if st.session_state.saved_models:
            st.markdown("**Saved Models:**")
            models_data = []
            for name, data in st.session_state.saved_models.items():
                models_data.append({
                    'Name': name,
                    'Parameters': ', '.join(data['param_names'][:3]) + ('...' if len(data['param_names']) > 3 else ''),
                    'Variables': data.get('auto_detected_vars', 'Unknown'),
                    'Created': data['created']
                })
            
            models_df = pd.DataFrame(models_data)
            st.dataframe(models_df, use_container_width=True)
            
            # Delete model
            model_to_delete = st.selectbox("Delete model:", [""] + list(st.session_state.saved_models.keys()), key="delete_model")
            if model_to_delete and st.button("🗑️ Delete Model", key="delete_model_btn"):
                del st.session_state.saved_models[model_to_delete]
                st.success(f"Deleted model '{model_to_delete}'")
                st.rerun()
        
        # Display saved bounds
        if st.session_state.saved_bounds:
            st.markdown("**Saved Bounds:**")
            bounds_data = []
            for name, data in st.session_state.saved_bounds.items():
                bounds_data.append({
                    'Name': name,
                    'Parameters': ', '.join(data['param_names'][:3]) + ('...' if len(data['param_names']) > 3 else ''),
                    'Created': data['created']
                })
            
            bounds_df = pd.DataFrame(bounds_data)
            st.dataframe(bounds_df, use_container_width=True)
            
            # Delete bounds
            bounds_to_delete = st.selectbox("Delete bounds:", [""] + list(st.session_state.saved_bounds.keys()), key="delete_bounds")
            if bounds_to_delete and st.button("🗑️ Delete Bounds", key="delete_bounds_btn"):
                del st.session_state.saved_bounds[bounds_to_delete]
                st.success(f"Deleted bounds '{bounds_to_delete}'")
                st.rerun()
    
    with tab4:
        st.markdown("### Import/Export Library")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Export Library**")
            if st.session_state.saved_models or st.session_state.saved_bounds:
                json_data = export_model_library()
                st.download_button(
                    label="📥 Download Model Library",
                    data=json_data,
                    file_name=f"mODEl_library_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                with st.expander("Preview Export Data"):
                    st.code(json_data[:500] + "..." if len(json_data) > 500 else json_data, language="json")
            else:
                st.info("No models or bounds to export")
        
        with col2:
            st.markdown("**Import Library**")
            uploaded_library = st.file_uploader(
                "Upload model library file:",
                type=['json'],
                help="Upload a previously exported model library",
                key="import_library_file"
            )
            
            if uploaded_library:
                try:
                    json_data = uploaded_library.read().decode('utf-8')
                    
                    # Preview import data
                    with st.expander("Preview Import Data"):
                        preview_data = json.loads(json_data)
                        st.write(f"**Models:** {len(preview_data.get('models', {}))}")
                        st.write(f"**Bounds:** {len(preview_data.get('bounds', {}))}")
                        if 'exported' in preview_data:
                            st.write(f"**Exported:** {preview_data['exported']}")
                    
                    if st.button("📤 Import Library", key="import_library_btn"):
                        import_model_library(json_data)
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")


def get_model_library_quick_load():
    """Get quick load options for sidebar"""
    quick_options = {}
    
    if st.session_state.saved_models:
        quick_options['models'] = list(st.session_state.saved_models.keys())
    
    if st.session_state.saved_bounds:
        quick_options['bounds'] = list(st.session_state.saved_bounds.keys())
    
    return quick_options 