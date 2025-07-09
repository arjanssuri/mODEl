"""
mODEl Bootstrap Analysis Module
Handles bootstrap analysis for parameter uncertainty estimation
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from scipy.optimize import minimize
from datetime import datetime
import time
from model_fitting import create_ode_function
import io


def calculate_adaptive_bins(data, method="Auto (Adaptive)"):
    """Calculate optimal number of bins for histogram based on different methods"""
    n = len(data)
    
    if method == "Sturges":
        # Sturges' rule: k = ceil(log2(n) + 1)
        bins = int(np.ceil(np.log2(n) + 1))
    elif method == "Scott":
        # Scott's rule: h = 3.5 * std(data) / n^(1/3)
        h = 3.5 * np.std(data) / (n**(1/3))
        bins = int(np.ceil((np.max(data) - np.min(data)) / h))
    elif method == "Freedman-Diaconis":
        # Freedman-Diaconis rule: h = 2 * IQR / n^(1/3)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            h = 2 * iqr / (n**(1/3))
            bins = int(np.ceil((np.max(data) - np.min(data)) / h))
        else:
            bins = 10  # fallback
    elif method == "Rice":
        # Rice rule: k = 2 * n^(1/3)
        bins = int(np.ceil(2 * (n**(1/3))))
    else:  # Auto (Adaptive)
        # Use numpy's auto method which chooses the best from several methods
        bins = 'auto'
    
    # Ensure reasonable bounds
    if isinstance(bins, int):
        bins = max(5, min(bins, 100))  # Between 5 and 100 bins
    
    return bins


def format_bin_info(data, bins_used):
    """Format information about the bins used"""
    if isinstance(bins_used, str):
        return f"Auto-selected optimal bins"
    else:
        range_val = np.max(data) - np.min(data)
        bin_width = range_val / bins_used if bins_used > 0 else 0
        return f"{bins_used} bins (width: {bin_width:.3e})"


def render_bootstrap_analysis_tab():
    """Render the Bootstrap Analysis tab content"""
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
                                                value=st.session_state.bootstrap_settings['n_samples'], 
                                                min_value=10, max_value=2000,
                                                help="Maximum 2000 samples allowed")
            
            # Show warning and confirmation for high sample counts
            show_confirmation = False
            if n_bootstrap_samples > 500:
                st.warning(f"⚠️ **High Sample Count Warning**\n\n"
                          f"You've selected {n_bootstrap_samples} samples. "
                          f"This may take significant computational time!")
                
                if n_bootstrap_samples > 1000:
                    st.error(f"🚨 **Very High Sample Count** ({n_bootstrap_samples} samples)\n\n"
                            f"This could take a very long time to complete. Consider reducing the number.")
                
                show_confirmation = True
            
            bootstrap_method = st.selectbox("Bootstrap Method", 
                                          ["Residual Resampling", "Parametric Bootstrap"],
                                          index=["Residual Resampling", "Parametric Bootstrap"].index(st.session_state.bootstrap_settings['method']))
            
            confidence_level = st.selectbox("Confidence Level", [90, 95, 99], 
                                           index=[90, 95, 99].index(st.session_state.bootstrap_settings['confidence_level']))
            
            # Logging frequency settings
            st.subheader("📝 Logging Settings")
            log_every = st.number_input("Log Progress Every X Samples", 
                                       value=10, min_value=1, max_value=100,
                                       help="How often to display progress updates")
            
            show_detailed_logs = st.checkbox("Show Parameter Values in Log", 
                                            value=True,
                                            help="Display parameter values for each logged sample")
            
            # Histogram binning settings
            st.subheader("📊 Distribution Plot Settings")
            bin_method = st.selectbox("Histogram Binning Method", 
                                    ["Auto (Adaptive)", "Sturges", "Scott", "Freedman-Diaconis", "Rice", "Fixed (20 bins)"],
                                    index=0,
                                    help="Method for determining optimal number of histogram bins")
            
            if bin_method == "Fixed (20 bins)":
                custom_bins = st.number_input("Number of Bins", 
                                            value=20, min_value=5, max_value=100,
                                            help="Fixed number of bins for histograms")
            else:
                st.info(f"**{bin_method}**: Automatically determines optimal bin count based on data distribution")
                custom_bins = None
            
            # Update session state with current selections
            st.session_state.bootstrap_settings['n_samples'] = n_bootstrap_samples
            st.session_state.bootstrap_settings['method'] = bootstrap_method
            st.session_state.bootstrap_settings['confidence_level'] = confidence_level
            
            # Run Bootstrap Analysis Button with confirmation
            if show_confirmation:
                st.markdown("### ⚠️ Confirmation Required")
                confirm_high_samples = st.checkbox(f"Yes, I want to run {n_bootstrap_samples} bootstrap samples", 
                                                  value=False,
                                                  help="Check this box to confirm you want to proceed with this many samples")
                run_button_disabled = not confirm_high_samples
                button_label = f"🎯 Run Bootstrap Analysis ({n_bootstrap_samples} samples)"
            else:
                run_button_disabled = False
                button_label = "🎯 Run Bootstrap Analysis"
            
        with col2:
            st.subheader("📝 Real-Time Analysis Log")
            
            # Initialize log containers that persist
            if 'bootstrap_log_container' not in st.session_state:
                st.session_state.bootstrap_log_container = None
            if 'bootstrap_timer_container' not in st.session_state:
                st.session_state.bootstrap_timer_container = None
            
            # Create permanent containers for log and timer
            timer_container = st.empty()
            log_container = st.empty()
            
            # Show initial state
            if not hasattr(st.session_state, 'bootstrap_in_progress') or not st.session_state.bootstrap_in_progress:
                timer_container.info("⏱️ **Ready to start** - Click 'Run Bootstrap Analysis'")
                log_container.text_area("Bootstrap Log", 
                                       value="🔄 Waiting for analysis to start...\n\nConfiguration:\n- Click the button to begin\n- Real-time progress will appear here\n- SSR and parameter values will be logged", 
                                       height=300, 
                                       disabled=True,
                                       key="initial_log")
        
        # Back to left column for the button
        with col1:
            if st.button(button_label, type="primary", disabled=run_button_disabled):
                # Set analysis in progress flag
                st.session_state.bootstrap_in_progress = True
                
                # Start timer
                start_time = time.time()
                
                # Initialize logging in right column
                log_messages = []
                max_log_lines = 50  # Maximum lines to keep in log
                
                def add_log_message(message, sample_num=None, ssr=None, params=None):
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    if sample_num is not None and ssr is not None:
                        # Format sample-specific log entry
                        log_entry = f"[{timestamp}] Sample {sample_num}"
                        if params and show_detailed_logs:
                            param_str = ", ".join([f"{name}={val:.2e}" for name, val in zip(st.session_state.param_names, params)])
                            log_entry += f"\n    SSR: {ssr:.3e} | {param_str}"
                        else:
                            log_entry += f" | SSR: {ssr:.3e}"
                    else:
                        # Regular log message
                        log_entry = f"[{timestamp}] {message}"
                    
                    log_messages.append(log_entry)
                    
                    # Truncate log if it gets too long
                    if len(log_messages) > max_log_lines:
                        # Keep first few and last portion
                        keep_first = 5
                        keep_last = max_log_lines - keep_first - 1
                        log_messages[:] = (log_messages[:keep_first] + 
                                         [f"... [Truncated {len(log_messages) - keep_first - keep_last} entries] ..."] +
                                         log_messages[-keep_last:])
                    
                    # Force real-time update of log display
                    log_text = "\n".join(log_messages)
                    log_container.text_area("📝 Bootstrap Analysis Log", 
                                           value=log_text, 
                                           height=300, 
                                           key=f"bootstrap_log_{len(log_messages)}_{time.time()}")
                
                def update_timer():
                    elapsed = time.time() - start_time
                    minutes = int(elapsed // 60)
                    seconds = int(elapsed % 60)
                    timer_container.info(f"⏱️ **Elapsed Time:** {minutes:02d}:{seconds:02d}")
                
                try:
                    add_log_message(f"🚀 Starting bootstrap analysis")
                    add_log_message(f"📊 {n_bootstrap_samples} samples | Method: {bootstrap_method}")
                    add_log_message(f"🎯 Confidence Level: {confidence_level}% | Log every {log_every} samples")
                    
                    # Create ODE function for bootstrap
                    ode_func = create_ode_function(st.session_state.param_names, st.session_state.ode_system)
                    add_log_message("✅ ODE function compiled successfully")
                    
                    # Get fitted parameters
                    fitted_params = [st.session_state.fit_results['params'][p] for p in st.session_state.param_names]
                    fitted_params_str = ", ".join([f"{name}={val:.2e}" for name, val in zip(st.session_state.param_names, fitted_params)])
                    add_log_message(f"📈 Original fit: {fitted_params_str}")
                    
                    # Prepare datasets for bootstrap
                    all_times = []
                    for data in st.session_state.datasets.values():
                        all_times.extend(data['time'].values)
                    unique_times = sorted(set(all_times))
                    t_data = np.array(unique_times)
                    add_log_message(f"⚙️ Time grid: {len(t_data)} points, {len(st.session_state.datasets)} datasets")
                    
                    # Calculate residuals for the original fit
                    sol_orig = odeint(ode_func, st.session_state.initial_conditions, t_data, 
                                    args=tuple(fitted_params))
                    
                    original_residuals = {}
                    for dataset_name, data in st.session_state.datasets.items():
                        var_idx = st.session_state.dataset_mapping[dataset_name]
                        model_vals = np.interp(data['time'], t_data, sol_orig[:, var_idx])
                        original_residuals[dataset_name] = data['value'] - model_vals
                    
                    add_log_message(f"📊 Calculated residuals for all datasets")
                    add_log_message("━" * 50)
                    add_log_message("🔄 Starting bootstrap sampling...")
                    
                    # Bootstrap analysis
                    bootstrap_params = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(n_bootstrap_samples):
                        try:
                            # Update timer every few iterations
                            if i % 5 == 0:
                                update_timer()
                            
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
                                if (hasattr(st.session_state, 'parsed_bounds') and 
                                    st.session_state.parsed_bounds and 
                                    param in st.session_state.parsed_bounds):
                                    # Ensure bounds are tuples, not arrays
                                    bound_val = st.session_state.parsed_bounds[param]
                                    if isinstance(bound_val, (list, np.ndarray)):
                                        bounds.append((float(bound_val[0]), float(bound_val[1])))
                                    elif isinstance(bound_val, tuple) and len(bound_val) == 2:
                                        bounds.append((float(bound_val[0]), float(bound_val[1])))
                                    else:
                                        # Fallback to default bounds
                                        fitted_val = float(st.session_state.fit_results['params'][param])
                                        bounds.append((fitted_val * 0.01, fitted_val * 100))
                                else:
                                    # Default bounds around fitted value
                                    fitted_val = float(st.session_state.fit_results['params'][param])
                                    bounds.append((fitted_val * 0.01, fitted_val * 100))
                            
                            # Optimize bootstrap sample with error handling
                            try:
                                result = minimize(bootstrap_objective, fitted_params, 
                                                method=st.session_state.optimization_settings['method'], 
                                                bounds=bounds,
                                                options={'maxiter': 100})  # Limit iterations for bootstrap
                                
                                if result.success and np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
                                    bootstrap_params.append(result.x)
                                    
                                    # Log progress according to user setting with nice formatting
                                    if (i + 1) % log_every == 0:
                                        elapsed = time.time() - start_time
                                        rate = (i + 1) / elapsed if elapsed > 0 else 0
                                        eta = (n_bootstrap_samples - i - 1) / rate if rate > 0 else 0
                                        
                                        success_rate = len(bootstrap_params) / (i + 1) * 100
                                        
                                        log_msg = f"✅ {i+1}/{n_bootstrap_samples} ({success_rate:.1f}% success, {rate:.1f}/sec, ETA: {eta:.0f}s)"
                                        add_log_message(log_msg, sample_num=i+1, ssr=result.fun, params=result.x)
                                else:
                                    if (i + 1) % log_every == 0:
                                        reason = "non-finite result" if not np.isfinite(result.fun) else "did not converge"
                                        add_log_message(f"❌ Sample {i+1}/{n_bootstrap_samples} FAILED - {reason}")
                            
                            except Exception as opt_error:
                                if (i + 1) % log_every == 0:
                                    add_log_message(f"❌ Sample {i+1}/{n_bootstrap_samples} OPTIMIZATION ERROR: {str(opt_error)[:60]}...")
                        
                        except Exception as e:
                            # Log failed samples
                            if (i + 1) % log_every == 0:
                                add_log_message(f"💥 Sample {i+1}/{n_bootstrap_samples} SETUP ERROR: {str(e)[:60]}...")
                        
                        # Update progress bar and status
                        progress = (i + 1) / n_bootstrap_samples
                        progress_bar.progress(progress)
                        status_text.text(f"Processing sample {i+1}/{n_bootstrap_samples} "
                                       f"({len(bootstrap_params)} successful)")
                        
                        # Force a small delay to allow UI updates - this is crucial for real-time updates
                        time.sleep(0.01)
                    
                    # Final timer update
                    update_timer()
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    total_elapsed = time.time() - start_time
                    add_log_message("━" * 50)
                    add_log_message(f"🏁 Analysis completed in {total_elapsed:.1f} seconds")
                    add_log_message(f"📊 Final success rate: {len(bootstrap_params)}/{n_bootstrap_samples} ({100*len(bootstrap_params)/n_bootstrap_samples:.1f}%)")
                    
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
                        
                        add_log_message("📈 Statistical analysis completed")
                        
                        # Store results
                        st.session_state.bootstrap_results = {
                            'n_samples': n_bootstrap_samples,
                            'method': bootstrap_method,
                            'confidence_level': confidence_level,
                            'successful_samples': len(bootstrap_params),
                            'stats': bootstrap_stats,
                            'elapsed_time': total_elapsed
                        }
                        
                        add_log_message("🎉 Bootstrap analysis completed successfully!")
                        st.success(f"✅ Bootstrap analysis completed! "
                                 f"({len(bootstrap_params)}/{n_bootstrap_samples} successful samples) "
                                 f"in {total_elapsed:.1f} seconds")
                    
                    else:
                        add_log_message("❌ No successful samples obtained")
                        st.error("❌ Bootstrap analysis failed - no successful samples")
                    
                except Exception as e:
                    elapsed = time.time() - start_time
                    add_log_message(f"💥 FATAL ERROR after {elapsed:.1f}s: {str(e)}")
                    st.error(f"❌ Bootstrap analysis error: {str(e)}")
                    st.exception(e)
                finally:
                    # Reset progress flag
                    st.session_state.bootstrap_in_progress = False

        # Bootstrap Results Section (full width)
        st.markdown("---")
        st.subheader("Bootstrap Results")
        
        if st.session_state.bootstrap_results:
            # Display elapsed time
            if 'elapsed_time' in st.session_state.bootstrap_results:
                elapsed = st.session_state.bootstrap_results['elapsed_time']
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                st.info(f"⏱️ **Analysis completed in:** {minutes:02d}:{seconds:02d}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
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
            
            with col2:
                # Export bootstrap results
                if st.button("📥 Export Bootstrap Results"):
                    elapsed_time = st.session_state.bootstrap_results.get('elapsed_time', 0)
                    minutes = int(elapsed_time // 60)
                    seconds = int(elapsed_time % 60)
                    
                    # Create summary report
                    bootstrap_export = f"""# mODEl Bootstrap Analysis Results - Dobrovolny Lab TCU
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- Bootstrap samples: {st.session_state.bootstrap_results['n_samples']}
- Successful samples: {st.session_state.bootstrap_results['successful_samples']}
- Success rate: {100*st.session_state.bootstrap_results['successful_samples']/st.session_state.bootstrap_results['n_samples']:.1f}%
- Confidence level: {st.session_state.bootstrap_results['confidence_level']}%
- Method: {st.session_state.bootstrap_results['method']}
- Computation time: {minutes:02d}:{seconds:02d}

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
                    
                    # Create distributions CSV data
                    distributions_data = []
                    for i in range(st.session_state.bootstrap_results['successful_samples']):
                        row = {'Sample': i + 1}
                        for param in st.session_state.param_names:
                            row[param] = st.session_state.bootstrap_results['stats'][param]['values'][i]
                        distributions_data.append(row)
                    
                    distributions_df = pd.DataFrame(distributions_data)
                    csv_buffer = io.StringIO()
                    distributions_df.to_csv(csv_buffer, index=False)
                    distributions_csv = csv_buffer.getvalue()
                    
                    # Create complete export package
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📄 Download Summary Report",
                            data=bootstrap_export,
                            file_name=f"mODEl_bootstrap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/plain"
                        )
                        
                        st.info("**Summary Report** contains:\n- Configuration details\n- Statistical summary\n- Confidence intervals")
                    
                    with col2:
                        st.download_button(
                            label="📊 Download Parameter Distributions",
                            data=distributions_csv,
                            file_name=f"mODEl_bootstrap_distributions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        st.info("**Distributions CSV** contains:\n- All bootstrap parameter values\n- Ready for plotting/analysis\n- Compatible with R, Python, Excel")
                    
                    # Show preview of distributions data
                    with st.expander("📋 Preview Parameter Distributions Data"):
                        st.markdown(f"**{st.session_state.bootstrap_results['successful_samples']} bootstrap samples** × **{len(st.session_state.param_names)} parameters**")
                        st.dataframe(distributions_df.head(10), use_container_width=True)
                        if len(distributions_df) > 10:
                            st.info(f"Showing first 10 rows of {len(distributions_df)} total samples")
                        
                        # Basic statistics preview
                        st.markdown("**Quick Statistics:**")
                        stats_preview = distributions_df.describe()
                        st.dataframe(stats_preview, use_container_width=True)
        else:
            st.info("Run bootstrap analysis to see results here.")

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
                    
                    # Determine bins based on selected method
                    if bin_method == "Fixed (20 bins)":
                        bins_used = custom_bins
                    else:
                        bins_used = calculate_adaptive_bins(values, bin_method)
                    
                    # Add histogram
                    fig.add_trace(
                        go.Histogram(
                            x=values,
                            name=f'{param} Distribution',
                            marker_color='skyblue',
                            opacity=0.7,
                            showlegend=False,
                            nbinsx=bins_used if isinstance(bins_used, int) else None,
                            autobinx=True if bins_used == 'auto' else False
                        ),
                        row=row, col=col
                    )
                    
                    # Add vertical lines for statistics
                    y_max = np.histogram(values, bins=bins_used)[0].max()
                    
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
                
                fig.update_layout(
                    height=400*rows,
                    title_text=f"Parameter Distribution Analysis - {bin_method} Binning",
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display binning information
                with st.expander("📊 Binning Information"):
                    st.markdown("**Adaptive Binning Results:**")
                    for param in st.session_state.param_names:
                        values = st.session_state.bootstrap_results['stats'][param]['values']
                        if bin_method == "Fixed (20 bins)":
                            bins_info = f"{custom_bins} bins (fixed)"
                        else:
                            bins_used = calculate_adaptive_bins(values, bin_method)
                            bins_info = format_bin_info(values, bins_used)
                        st.markdown(f"- **{param}**: {bins_info}")
                    
                    st.info(f"**Method**: {bin_method}")
                    if bin_method != "Fixed (20 bins)":
                        method_descriptions = {
                            "Auto (Adaptive)": "NumPy's automatic method - selects optimal approach based on data",
                            "Sturges": "k = ceil(log₂(n) + 1) - Good for normal distributions",
                            "Scott": "Based on data standard deviation - Good for smooth distributions", 
                            "Freedman-Diaconis": "Based on interquartile range - Robust to outliers",
                            "Rice": "k = 2 × n^(1/3) - Simple cube root rule"
                        }
                        if bin_method in method_descriptions:
                            st.markdown(f"**Description**: {method_descriptions[bin_method]}")
                
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
                    
                    # Determine bins based on selected method
                    if bin_method == "Fixed (20 bins)":
                        bins_used = custom_bins
                    else:
                        bins_used = calculate_adaptive_bins(values, bin_method)
                    
                    n_hist, bins_hist, patches = ax.hist(values, bins=bins_used, alpha=0.7, color='skyblue', edgecolor='black')
                    
                    # Add bin count to title
                    actual_bins = len(bins_hist) - 1 if hasattr(bins_hist, '__len__') else bins_used
                    ax.set_title(f'{param} ({actual_bins} bins)')
                    
                    # Add vertical lines for statistics
                    ax.axvline(st.session_state.fit_results['params'][param], 
                              color='red', linestyle='--', linewidth=2, label='Original')
                    ax.axvline(st.session_state.bootstrap_results['stats'][param]['mean'], 
                              color='green', linestyle='-', linewidth=2, label='Bootstrap Mean')
                    ax.axvline(st.session_state.bootstrap_results['stats'][param]['ci_lower'], 
                              color='orange', linestyle=':', linewidth=2, label=f'{confidence_level}% CI')
                    ax.axvline(st.session_state.bootstrap_results['stats'][param]['ci_upper'], 
                              color='orange', linestyle=':', linewidth=2)
                    
                    ax.set_xlabel('Parameter Value')
                    ax.set_ylabel('Frequency')
                    if i == 0:
                        ax.legend()
                
                # Remove empty subplots
                for i in range(n_params, rows * cols):
                    row, col = i // cols, i % cols
                    if rows == 1:
                        if cols > 1:
                            fig.delaxes(axes[col])
                    else:
                        if cols > 1:
                            fig.delaxes(axes[row, col])
                        else:
                            fig.delaxes(axes[row])
                
                plt.suptitle(f"Parameter Distribution Analysis - {bin_method} Binning")
                plt.tight_layout()
                st.pyplot(fig) 