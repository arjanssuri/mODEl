"""
mODEl: Advanced ODE Model Fitting Tool
Clean, modular main application file

by Dobrovolny Lab, Texas Christian University
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from datetime import datetime

# Import our custom modules
from utils import initialize_session_state, get_completion_status
from model_fitting import (
    run_model_fitting, detect_state_variables, extract_parameter_names, 
    create_ode_function
)
from data_management import (
    upload_individual_file_ui, organize_batch_files, process_batch_files,
    load_job_into_workspace, save_workspace_to_job, get_jobs_summary,
    export_jobs_summary
)
from ode_examples import ODE_EXAMPLES
from ode_definition import render_ode_definition_tab
from parameter_fitting import render_parameter_fitting_tab
from results_analysis import render_results_analysis_tab
from bootstrap_analysis import render_bootstrap_analysis_tab

# Set page configuration
st.set_page_config(
    page_title="mODEl - ODE Model Fitting by Dobrovolny Lab TCU",
    page_icon="🧮",
    layout="wide"
)

# Custom CSS and JavaScript
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
    .bootstrap-warning {
        background-color: #d1ecf1;
        border: 1px solid #b8daff;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        color: #0c5460;
    }
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
    .stButton > button[kind="primary"] {
        background-color: #0080FF !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0066CC !important;
        color: white !important;
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
    kbd {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 3px;
        padding: 2px 4px;
        font-size: 11px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    }
</style>

<script>
// Keyboard shortcut for model fitting (Cmd + . or Ctrl + .)
document.addEventListener('keydown', function(event) {
    if ((event.metaKey || event.ctrlKey) && event.key === '.') {
        event.preventDefault();
        
        const buttons = document.querySelectorAll('button');
        let sidebarButton = null;
        
        buttons.forEach(button => {
            if (button.textContent.includes('Quick Model Fitting')) {
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
        }
        
        return false;
    }
});
</script>
""", unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

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

# Sidebar for configuration and quick actions
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Optimization settings
    opt_method = st.selectbox(
        "Optimization Method",
        ["L-BFGS-B", "Nelder-Mead", "SLSQP", "Powell", "TNC", "Differential Evolution"],
        index=["L-BFGS-B", "Nelder-Mead", "SLSQP", "Powell", "TNC", "Differential Evolution"].index(st.session_state.optimization_settings['method']),
        help="Select the optimization algorithm for parameter fitting"
    )
    st.session_state.optimization_settings['method'] = opt_method
    
    # Convergence settings
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
    
    # Visualization settings
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
    
    # Progress tracker
    st.subheader("📋 Progress Tracker")
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
    
    # Quick fit button
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

# Tab 1: Data Upload with Batch Processing
with tab1:
    st.header("Upload Experimental Data to mODEl")
    
    # Data upload method selection
    upload_method = st.radio(
        "Choose upload method:",
        ["Individual Files", "Batch Folder Upload"],
        help="Upload individual files or process multiple files as batch jobs"
    )
    
    if upload_method == "Individual Files":
        upload_individual_file_ui()
    
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
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Job Organization")
                job_organization = st.radio(
                    "How to organize files into jobs:",
                    ["Create separate job for each file", "Group by filename prefix", "Group all files into one job"],
                    help="Choose how to organize the uploaded files into modeling jobs"
                )
                
                # Organize files and display structure
                organized_jobs = organize_batch_files(uploaded_files, job_organization)
                
                st.write("**Proposed Job Organization:**")
                for job_name, files in organized_jobs.items():
                    st.write(f"- **{job_name}**: {len(files)} files")
                    for file in files:
                        st.write(f"  - {file.name}")
            
            with col2:
                st.subheader("Process Batch Jobs")
                
                if st.button("🚀 Create Batch Jobs", type="primary"):
                    jobs_created, jobs_failed = process_batch_files(organized_jobs)
                    
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
                            if load_job_into_workspace(selected_job):
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
                        if save_workspace_to_job():
                            st.success(f"✅ Saved current state with all advanced settings to job '{st.session_state.active_job}'")
                else:
                    st.warning("No active job selected")
            
            # Jobs summary table
            st.subheader("📊 Jobs Summary")
            summary_df = get_jobs_summary()
            
            if not summary_df.empty:
                st.dataframe(summary_df, use_container_width=True)
                
                # Export jobs summary
                if st.button("📥 Export Jobs Summary"):
                    csv_data = export_jobs_summary()
                    st.download_button(
                        label="Download Jobs Summary CSV",
                        data=csv_data,
                        file_name=f"mODEl_batch_jobs_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

# Tab 2: ODE Definition
with tab2:
    render_ode_definition_tab()

# Tab 3: Parameter Fitting
with tab3:
    render_parameter_fitting_tab()

# Tab 4: Results Analysis
with tab4:
    render_results_analysis_tab()

# Tab 5: Bootstrap Analysis
with tab5:
    render_bootstrap_analysis_tab()

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