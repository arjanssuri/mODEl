# 🧮 mODEl: Advanced ODE Model Fitting Tool

**by Dobrovolny Lab, Texas Christian University**

A comprehensive Streamlit application for fitting ordinary differential equation (ODE) models to experimental data, with advanced features for uncertainty quantification, multi-dataset analysis, and batch processing.

## 🚀 New Features

### 🔄 Version 2.0 Enhancements
- **📁 Batch Folder Upload**: Upload multiple files and organize them into separate modeling jobs
- **🎯 Smart Initial Conditions**: Set initial conditions automatically from first data values
- **⚙️ Improved Code Organization**: Scripts moved to `src/`, examples organized in `examples/`
- **🔁 Job Management**: Process multiple datasets as independent jobs with state preservation

## ✨ Core Features

### Advanced Capabilities
- **Multi-dataset Upload**: Upload and analyze multiple related datasets simultaneously
- **Custom ODE Definition**: Define complex multi-variable ODE systems using Python syntax
- **Multiple Optimization Algorithms**: L-BFGS-B, Nelder-Mead, SLSQP, Powell, TNC, Differential Evolution
- **Bootstrap Analysis**: Comprehensive uncertainty quantification for parameter estimates
- **Interactive Visualization**: Both Matplotlib and Plotly visualizations with customizable styling
- **Results Export**: Download complete analysis packages with parameters, statistics, and plots

### Enhanced Features
- **Batch Processing**: Upload multiple files and process them as separate jobs
- **Smart Initial Conditions**: Automatically set initial conditions from first data values
- **Multi-start Optimization**: Robust parameter fitting with multiple initial guesses
- **Relative vs Absolute Error**: Choose between error metrics for different data types
- **Weighted Least Squares**: Assign different weights to different datasets
- **Log Transformations**: Handle positive-only data with log transforms
- **Phase Portrait Analysis**: Visualize 2D system dynamics
- **Parameter Distributions**: Visualize uncertainty in parameter estimates
- **Confidence Intervals**: Calculate and display parameter confidence intervals

## 📁 Project Structure

```
mODEl/
├── src/                     # Application source code
│   ├── app.py              # Main Streamlit application
│   ├── ode_examples.py     # Built-in ODE examples and helper functions
│   └── run_app.sh          # Local run script
├── examples/               # Sample datasets and documentation
│   ├── README.md           # Guide to using sample datasets
│   ├── sample_data.txt     # Simple exponential decay data
│   ├── viral_load_sample.txt      # Viral dynamics - viral load
│   ├── interferon_sample.txt      # Viral dynamics - interferon
│   ├── logistic_growth_sample.txt # Population growth data
│   ├── sir_susceptible_sample.txt # SIR model - susceptible population
│   ├── sir_infected_sample.txt    # SIR model - infected population
│   ├── predator_sample.txt        # Predator-prey - predator population
│   └── prey_sample.txt            # Predator-prey - prey population
├── requirements.txt        # Python dependencies
├── run_model.sh           # Main launcher script
├── README.md              # This file
└── QUICK_START.md         # Quick start guide
```

## 🛠️ Installation & Setup

### Option 1: Quick Start (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-repo/mODEl.git
cd mODEl

# Make launcher executable and run
chmod +x run_model.sh
./run_model.sh
```

### Option 2: Network Deployment
For running mODEl on a network (accessible from other computers):
```bash
# Make network launcher executable
chmod +x run_model_network.sh

# Run with interactive network configuration
./run_model_network.sh
```

### Option 3: Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run from src directory
cd src
streamlit run app.py
```

### Option 4: Custom Network Configuration
For advanced users who want to specify custom Streamlit options:
```bash
# Example: Run on specific IP and port
./run_model.sh --server.address 192.168.1.100 --server.port 8080

# Example: Run on all interfaces
./run_model.sh --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

## 📖 Usage Guide

### 🔄 Batch Processing Workflow

#### Individual File Processing
1. Go to **📁 Data Upload** tab
2. Choose "Individual Files" option
3. Upload files one by one with custom names
4. Configure ODE system and fit parameters

#### Batch Folder Processing
1. Go to **📁 Data Upload** tab
2. Choose "Batch Folder Upload" option
3. Upload multiple files simultaneously
4. Choose organization method:
   - **Separate job for each file**: Individual analysis for each file
   - **Group by filename prefix**: Group files with common prefixes (e.g., `experiment1_viral.txt`, `experiment1_interferon.txt`)
   - **Group all into one job**: Combine all files into a single analysis
5. Manage jobs in the batch jobs panel
6. Load jobs into workspace for analysis

### 🎯 Smart Initial Conditions

#### Manual Input (Traditional)
- Set initial conditions manually using number inputs
- Full control over starting values
- Good for theoretical starting points

#### Use First Data Values (New!)
- Automatically uses first value from each mapped dataset
- Perfect when your measurements start at t=0
- Requires dataset mapping to be configured first

#### Dataset-Specific First Values (New!)
- Choose which dataset provides the initial condition for each state variable
- Maximum flexibility for complex multi-dataset scenarios
- Ideal when different variables have different measurement starting points

### 📊 Data Upload & Management

**Supported Formats:**
- CSV and TXT files (tab, comma, semicolon, or space delimited)
- Required columns: 'time' and 'value' (case insensitive)
- Automatic delimiter detection and column mapping

**Data Validation:**
- Automatic validation of uploaded data
- Missing value detection and handling
- Data quality checks and warnings
- Interactive data visualization and statistics

### 🧬 ODE System Definition

Define your ODE system using Python syntax with full support for:
- **Multi-variable systems**: Use `y[0], y[1], y[2]...` for state variables
- **Parameter detection**: Automatically detects parameter names from equations
- **Built-in examples**: Choose from common ODE systems

**Example - Viral Dynamics Model:**
```python
# Viral dynamics with immune response
T, R, I, V, F = y
dTdt = -beta * T * V - gamma * T * F    # Target cells
dRdt = gamma * T * F - rho * R          # Resistant cells
dIdt = beta * T * V - delta * I         # Infected cells
dVdt = p * I - c * V                    # Virus
dFdt = I - alpha * F                    # Interferon
return [dTdt, dRdt, dIdt, dVdt, dFdt]
```

### 🔧 Advanced Parameter Configuration

**Individual Parameter Input:**
- Set bounds and initial guesses for each parameter individually
- Visual bounds validation and analysis

**Code-based Definition:**
- Define parameter bounds using Python dictionary syntax
- Load from configuration files
- Mathematical expressions and constants support
- Quick-load examples for common systems

### 📈 Model Fitting & Analysis

**Optimization Options:**
- Multiple algorithms with configurable settings
- Multi-start optimization for robust fitting
- Custom error metrics (relative/absolute)
- Dataset weighting for multi-dataset fitting

**Results Analysis:**
- Parameter estimates with bounds and confidence intervals
- Interactive model vs data visualizations
- Goodness of fit statistics (R², RMSE, AIC, BIC)
- Residual analysis and Q-Q plots
- Phase portrait analysis for 2D systems

### 🎯 Bootstrap Uncertainty Analysis

**Bootstrap Methods:**
- **Residual Resampling**: Resample residuals to estimate parameter uncertainty
- **Parametric Bootstrap**: Add noise based on residual variance

**Results:**
- Parameter confidence intervals (90%, 95%, 99%)
- Parameter distribution visualizations
- Correlation analysis between parameters
- Comprehensive uncertainty quantification

## 📚 Built-in Examples

The tool includes several pre-defined ODE systems with sample data:

1. **Exponential Decay**: Simple first-order decay
2. **Logistic Growth**: Population growth with carrying capacity  
3. **SIR Model**: Epidemiological susceptible-infected-recovered model
4. **Lotka-Volterra**: Predator-prey dynamics
5. **Chemical Reactions**: Multi-step reaction kinetics
6. **Enzyme Kinetics**: Michaelis-Menten kinetics
7. **Damped Oscillator**: Second-order damped harmonic oscillator
8. **Van der Pol Oscillator**: Nonlinear oscillator with limit cycle

## 🔬 Sample Datasets

Located in the `examples/` directory:
- **Single variable**: Exponential decay, logistic growth
- **Multi-variable**: Viral dynamics, SIR models, predator-prey
- **Format**: Tab-delimited with 'time' and 'value' columns
- **Documentation**: See `examples/README.md` for detailed usage instructions

## ⚡ Performance & Best Practices

### Optimization Tips
- Use **multi-start optimization** for complex parameter spaces
- Set **realistic parameter bounds** to improve convergence
- Use **smart initial conditions** from data when appropriate
- Consider **log transformations** for positive-only data
- Use **relative error** for data spanning multiple orders of magnitude

### Batch Processing Tips
- **Group related files** by filename prefix for efficient organization
- **Use descriptive filenames** to make job management easier
- **Save job states** frequently to preserve progress
- **Export results** for each job before switching

### Bootstrap Analysis Tips
- Start with **fewer samples** (50-100) for initial testing
- Use **residual resampling** for most applications
- **Parametric bootstrap** for well-characterized noise models
- Consider computational time for large datasets

## 🔧 Technical Details

### Numerical Methods
- **ODE Integration**: SciPy's `odeint` with adaptive step size
- **Optimization**: Multiple algorithms from SciPy optimize
- **Bootstrap**: Efficient numpy-based resampling
- **Statistics**: Robust statistical calculations

### Data Processing
- **Automatic validation** of uploaded data
- **Flexible file format** support (CSV, TXT)
- **Missing data handling** with interpolation options
- **Time series alignment** for multi-dataset analysis

### Job Management
- **Independent job processing** with state preservation
- **Flexible job organization** with multiple grouping options
- **Job state persistence** across sessions
- **Batch result export** and management

## 🔍 Troubleshooting

### Network Deployment Issues

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory: 'app.py'` when running on IP address

**Solution**: Use the provided network launcher scripts:
```bash
# Use the dedicated network script
./run_model_network.sh

# Or specify arguments to the main script
./run_model.sh --server.address YOUR_IP --server.port 8501
```

**Common Network Issues:**
- **Firewall blocking**: Ensure your firewall allows traffic on the specified port
- **Wrong working directory**: Always run scripts from the mODEl root directory
- **Permission issues**: Make sure scripts are executable (`chmod +x script_name.sh`)

### Application Issues

**Data Upload Problems:**
- Check file format (CSV/TXT with 'time' and 'value' columns)
- Ensure numeric data (no text in data columns)
- Verify delimiter detection (tab, comma, semicolon, or space)

**ODE Definition Errors:**
- Check Python syntax in ODE code
- Ensure parameter names match between ODE and bounds
- Verify number of initial conditions matches ODE variables

**Optimization Failures:**
- Try different optimization algorithms
- Check parameter bounds (ensure lower < upper)
- Use multi-start optimization for robust fitting
- Reduce tolerance or increase max iterations

### Performance Issues

**Slow Fitting:**
- Reduce dataset size for initial testing
- Use simpler ODE systems for validation
- Enable multi-start with fewer starts (5-10)

**Bootstrap Takes Too Long:**
- Start with fewer samples (50-100)
- Use simpler models for testing
- Consider using parametric bootstrap

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

**mODEl** is developed by the Dobrovolny Laboratory at Texas Christian University for mathematical modeling in biological systems.

Built with:
- [Streamlit](https://streamlit.io/) for the web interface
- [SciPy](https://scipy.org/) for numerical optimization
- [NumPy](https://numpy.org/) for numerical computations
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Matplotlib](https://matplotlib.org/) and [Plotly](https://plotly.com/) for visualization

## 🚀 Getting Started

1. **Quick Test**: Use the sample data in `examples/` to test the application
2. **Batch Processing**: Try uploading multiple files to see the job management features
3. **Smart Initial Conditions**: Test the automatic initial condition setting with your data
4. **Advanced Features**: Explore bootstrap analysis and multi-start optimization
5. **Network Deployment**: See `NETWORK_DEPLOYMENT.md` for running mODEl on a network
6. **Enhanced Batch Jobs**: See `BATCH_JOBS_FEATURES.md` for comprehensive batch processing with advanced analytics

**For detailed instructions, see `QUICK_START.md`**

---

**Happy modeling! 🚀**

For questions or support, please open an issue in the repository.
