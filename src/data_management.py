"""
mODEl Data Management Module
Handles data upload, batch processing, job management, and file operations
"""

import streamlit as st
import pandas as pd
import io
from datetime import datetime
from typing import Dict, List, Tuple, Optional


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read uploaded file with automatic delimiter detection using cache"""
    from utils import process_uploaded_data
    
    # Get file content for caching
    file_content = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer for other operations
    
    # Use cached processing
    return process_uploaded_data(file_content, uploaded_file.name)


def validate_data_file(data: pd.DataFrame) -> Tuple[bool, str, str, str]:
    """Validate uploaded data file and find time/value columns using cache"""
    from utils import validate_and_clean_data
    
    # Use cached validation
    is_valid, time_col, value_col, cleaned_data = validate_and_clean_data(data)
    
    if not is_valid:
        return False, "", "", f"Data must have 'time' and 'value' columns. Found columns: {', '.join(data.columns)}"
    
    return True, time_col, value_col, ""


def process_data_file(data: pd.DataFrame, time_col: str, value_col: str) -> Tuple[pd.DataFrame, str]:
    """Process and clean data file"""
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
        return data, "No valid data rows found after cleaning"
    
    mapping_info = ""
    if time_col != 'time' or value_col != 'value':
        mapping_info = f"Column mapping: '{time_col}' → time, '{value_col}' → value"
    
    return data, mapping_info


def upload_individual_file_ui():
    """UI for individual file upload"""
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
                data = read_uploaded_file(uploaded_file)
                is_valid, time_col, value_col, error_msg = validate_data_file(data)
                
                if not is_valid:
                    st.error(error_msg)
                    st.info("Acceptable column names:\n- Time: 'time', 't', 'times'\n- Value: 'value', 'val', 'values', 'concentration', 'conc', 'amount'")
                else:
                    # Show data preview before processing
                    with st.expander("📊 Data Preview"):
                        st.write(f"**File:** {uploaded_file.name}")
                        st.write(f"**Columns found:** {', '.join(data.columns)}")
                        st.write(f"**Detected time column:** {time_col}")
                        st.write(f"**Detected value column:** {value_col}")
                        st.write(f"**Number of rows:** {len(data)}")
                        
                        # Show first few rows
                        st.write("**First 5 rows:**")
                        st.dataframe(data.head(), use_container_width=True)
                        
                        # Show basic statistics
                        if time_col in data.columns and value_col in data.columns:
                            st.write("**Basic Statistics:**")
                            col_stats1, col_stats2 = st.columns(2)
                            with col_stats1:
                                st.write(f"**{time_col} (Time):**")
                                try:
                                    time_vals = pd.to_numeric(data[time_col], errors='coerce')
                                    st.write(f"- Range: {time_vals.min():.3f} to {time_vals.max():.3f}")
                                    st.write(f"- Points: {len(time_vals.dropna())}")
                                except:
                                    st.write("- Could not parse as numeric")
                            
                            with col_stats2:
                                st.write(f"**{value_col} (Value):**")
                                try:
                                    value_vals = pd.to_numeric(data[value_col], errors='coerce')
                                    st.write(f"- Range: {value_vals.min():.3f} to {value_vals.max():.3f}")
                                    st.write(f"- Mean: {value_vals.mean():.3f}")
                                except:
                                    st.write("- Could not parse as numeric")
                    
                    processed_data, mapping_info = process_data_file(data, time_col, value_col)
                    
                    if len(processed_data) == 0:
                        st.error("No valid data rows found after cleaning")
                    else:
                        # Show processing results
                        col_result1, col_result2 = st.columns(2)
                        with col_result1:
                            if st.button(f"✅ Load Dataset '{dataset_name}'", type="primary"):
                                st.session_state.datasets[dataset_name] = processed_data
                                st.success(f"✅ Dataset '{dataset_name}' loaded successfully! ({len(processed_data)} data points)")
                                
                                if mapping_info:
                                    st.info(mapping_info)
                                st.rerun()
                        
                        with col_result2:
                            st.info(f"**Ready to load:** {len(processed_data)} clean data points")
                            if mapping_info:
                                st.caption(mapping_info)
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.info("Make sure your file has columns for time and values, separated by tabs, commas, or spaces.")
        
        # Remove dataset button
        if st.session_state.datasets:
            st.markdown("---")
            st.subheader("🗑️ Remove Dataset")
            dataset_to_remove = st.selectbox("Remove Dataset", [""] + list(st.session_state.datasets.keys()))
            if dataset_to_remove and st.button("Remove Selected Dataset"):
                del st.session_state.datasets[dataset_to_remove]
                # Also remove from dataset mapping if it exists
                if dataset_to_remove in st.session_state.dataset_mapping:
                    del st.session_state.dataset_mapping[dataset_to_remove]
                st.success(f"Removed dataset '{dataset_to_remove}'")
                st.rerun()
    
    with col2:
        if st.session_state.datasets:
            st.subheader("📁 Loaded Datasets")
            for name, data in st.session_state.datasets.items():
                with st.expander(f"**{name}**"):
                    st.info(f"""
                    - **Rows:** {len(data)}
                    - **Time range:** {data['time'].min():.3f} - {data['time'].max():.3f}
                    - **Value range:** {data['value'].min():.3f} - {data['value'].max():.3f}
                    - **Mean value:** {data['value'].mean():.3f}
                    """)
                    
                    # Show mini plot
                    try:
                        import plotly.express as px
                        fig = px.scatter(data, x='time', y='value', 
                                       title=f"{name} Data Preview",
                                       height=200)
                        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        # Fallback to basic stats if plotly fails
                        st.write("**Sample data points:**")
                        st.dataframe(data.head(3), use_container_width=True)
        else:
            st.info("**No datasets loaded yet**\n\nUpload your experimental data files to get started with mODEl analysis.")


def organize_batch_files(uploaded_files, job_organization: str) -> Dict[str, List]:
    """Organize uploaded files into jobs based on organization method"""
    organized_jobs = {}
    
    if job_organization == "Create separate job for each file":
        # Each file becomes its own job
        for file in uploaded_files:
            job_name = file.name.rsplit('.', 1)[0]  # Remove extension
            organized_jobs[job_name] = [file]
    
    elif job_organization == "Group by filename prefix":
        # Group files by common prefix (before underscore or dash)
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
    
    return organized_jobs


def process_batch_files(organized_jobs: Dict[str, List]) -> Tuple[int, int]:
    """Process organized files into batch jobs"""
    jobs_created = 0
    jobs_failed = 0
    
    for job_name, files in organized_jobs.items():
        try:
            # Process files for this job
            job_datasets = {}
            
            for file in files:
                # Read and process each file
                try:
                    data = read_uploaded_file(file)
                    is_valid, time_col, value_col, error_msg = validate_data_file(data)
                    
                    if is_valid:
                        processed_data, _ = process_data_file(data, time_col, value_col)
                        
                        if len(processed_data) > 0:
                            dataset_name = file.name.rsplit('.', 1)[0]
                            job_datasets[dataset_name] = processed_data
                
                except Exception as e:
                    st.warning(f"Could not process file {file.name}: {str(e)}")
            
            # Create job if datasets were successfully processed
            if job_datasets:
                create_batch_job(job_name, job_datasets)
                jobs_created += 1
            else:
                jobs_failed += 1
        
        except Exception as e:
            st.error(f"Error creating job {job_name}: {str(e)}")
            jobs_failed += 1
    
    return jobs_created, jobs_failed


def create_batch_job(job_name: str, job_datasets: Dict[str, pd.DataFrame]):
    """Create a new batch job with datasets"""
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


def load_job_into_workspace(selected_job: str):
    """Load a batch job into the main workspace"""
    if selected_job in st.session_state.batch_jobs:
        job_data = st.session_state.batch_jobs[selected_job]
        
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
        
        return True
    return False


def save_workspace_to_job():
    """Save current workspace state back to the active job"""
    if st.session_state.active_job and st.session_state.active_job in st.session_state.batch_jobs:
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
        
        return True
    return False


def get_jobs_summary() -> pd.DataFrame:
    """Get summary of all batch jobs"""
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
    
    return pd.DataFrame(job_summary) if job_summary else pd.DataFrame()


def export_jobs_summary() -> str:
    """Export jobs summary as CSV string"""
    summary_df = get_jobs_summary()
    if not summary_df.empty:
        csv_buffer = io.StringIO()
        summary_df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
    return "" 