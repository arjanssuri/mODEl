# 🧮 Advanced ODE Model Fitting Tool

A comprehensive Streamlit application for fitting ordinary differential equation (ODE) models to experimental data, with advanced features for uncertainty quantification and multi-dataset analysis.

## 🚀 Features

### Core Capabilities

- **Multi-dataset Upload**: Upload and analyze multiple related datasets simultaneously
- **Custom ODE Definition**: Define complex multi-variable ODE systems using Python syntax
- **Advanced Optimization**: Multiple optimization algorithms (L-BFGS-B, Nelder-Mead, SLSQP, Powell, TNC, Differential Evolution)
- **Bootstrap Analysis**: Comprehensive uncertainty quantification for parameter estimates
- **Interactive Visualization**: Both Matplotlib and Plotly visualizations with customizable styling
- **Results Export**: Download complete analysis packages with parameters, statistics, and plots

### Advanced Features

- **Multi-start Optimization**: Robust parameter fitting with multiple initial guesses
- **Relative vs Absolute Error**: Choose between error metrics for different data types
- **Weighted Least Squares**: Assign different weights to different datasets
- **Log Transformations**: Handle positive-only data with log transforms
- **Phase Portrait Analysis**: Visualize 2D system dynamics
- **Parameter Distributions**: Visualize uncertainty in parameter estimates
- **Confidence Intervals**: Calculate and display parameter confidence intervals

## 📋 Requirements

```bash
streamlit==1.29.0
numpy==1.24.3
scipy==1.11.4
matplotlib==3.8.2
seaborn==0.13.0
pandas==2.1.4
plotly==5.18.0
sympy==1.12
```

## 🛠️ Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd mODEl
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the application**:

```bash
streamlit run app.py
```

Or use the provided script:

```bash
chmod +x run_app.sh
./run_app.sh
```

## 📖 Usage Guide

### 1. Data Upload

- **Multiple Datasets**: Upload separate files for different measured variables
- **Required Format**: Each file must contain 'time' and 'value' columns
- **Supported Formats**: CSV and TXT files (tab-delimited)
- **Data Validation**: Automatic validation and preview of uploaded data

**Example Data Format**:

```
time    value
0.0     10.0
0.5     8.2
1.0     6.7
1.5     5.5
```

### 2. ODE System Definition

Define your ODE system using Python syntax. The tool supports:

- **Multi-variable systems**: Use `y[0], y[1], y[2]...` for state variables
- **Parameter detection**: Automatically detects parameter names from your equations
- **Built-in examples**: Choose from common ODE systems (exponential decay, logistic growth, SIR models, etc.)

**Example - Viral Dynamics Model**:

```python
# Viral dynamics with immune response
T, R, I, V, F = y  # Unpack variables
dTdt = -beta * T * V - gamma * T * F    # Target cells
dRdt = gamma * T * F - rho * R          # Resistant cells
dIdt = beta * T * V - delta * I         # Infected cells
dVdt = p * I - c * V                    # Virus
dFdt = I - alpha * F                    # Interferon
return [dTdt, dRdt, dIdt, dVdt, dFdt]
```

### 3. Model Fitting

Configure parameter bounds and optimization settings:

- **Parameter Bounds**: Set realistic bounds for each parameter
- **Initial Guesses**: Provide starting values for optimization
- **Optimization Method**: Choose from multiple algorithms
- **Multi-start Option**: Use multiple random starting points for robust fitting
- **Error Metrics**: Choose between relative and absolute error

### 4. Results Analysis

Comprehensive analysis of fitting results:

- **Parameter Estimates**: View fitted parameters with bounds and initial guesses
- **Model Visualization**: Interactive plots comparing data and model predictions
- **Goodness of Fit**: R², RMSE, AIC, BIC statistics
- **Residual Analysis**: Residual plots and Q-Q plots for model validation
- **Phase Portraits**: For 2D systems, visualize phase space dynamics

### 5. Bootstrap Analysis

Uncertainty quantification through bootstrap resampling:

- **Residual Resampling**: Resample residuals to estimate parameter uncertainty
- **Parametric Bootstrap**: Add noise based on residual variance
- **Confidence Intervals**: Calculate 90%, 95%, or 99% confidence intervals
- **Parameter Distributions**: Visualize parameter uncertainty distributions

## 🎯 Advanced Examples

### Multi-Dataset Viral Dynamics

This example demonstrates fitting a complex viral dynamics model to multiple datasets:

```python
# Upload datasets:
# 1. viral_load.txt - Viral load measurements
# 2. interferon.txt - Interferon measurements

# Define ODE system:
T, R, I, V, F = y
dTdt = -beta * T * V - gamma * T * F
dRdt = gamma * T * F - rho * R
dIdt = beta * T * V - delta * I
dVdt = p * I - c * V
dFdt = I - alpha * F
return [dTdt, dRdt, dIdt, dVdt, dFdt]

# Map datasets to variables:
# viral_load -> y[3] (V)
# interferon -> y[4] (F)
```

### Parameter Bounds Configuration

```python
# Example parameter bounds for viral dynamics:
beta: [1e-6, 10.0]    # Infection rate
gamma: [1e-6, 10.0]   # Conversion to resistant state
rho: [1e-6, 10.0]     # Return from resistant state
delta: [1e-6, 10.0]   # Death rate of infected cells
p: [1e-1, 1e7]        # Virus production rate
c: [1e-6, 10.0]       # Virus clearance rate
alpha: [1e-6, 100.0]  # Interferon clearance rate
```

## 📊 Output and Export

### Results Package

The tool generates comprehensive results packages including:

- **fitted_parameters.txt**: All fitted parameters with timestamps
- **results_summary.csv**: Parameter summary table
- **dataset_info.txt**: Information about all uploaded datasets
- **parameter_distributions.png**: Bootstrap parameter distributions
- **model_fits.png**: Data vs model comparison plots

### Bootstrap Results

Bootstrap analysis provides:

- **Parameter means and standard deviations**
- **Confidence intervals at specified levels**
- **Parameter correlation analysis**
- **Distribution visualizations**

## 🔬 Built-in Examples

The tool includes several pre-defined ODE systems:

1. **Exponential Decay**: Simple first-order decay
2. **Logistic Growth**: Population growth with carrying capacity
3. **SIR Model**: Epidemiological susceptible-infected-recovered model
4. **Lotka-Volterra**: Predator-prey dynamics
5. **Chemical Reactions**: Multi-step reaction kinetics
6. **Enzyme Kinetics**: Michaelis-Menten kinetics
7. **Damped Oscillator**: Second-order damped harmonic oscillator
8. **Van der Pol Oscillator**: Nonlinear oscillator with limit cycle

## 🧪 Sample Data

The repository includes sample datasets for testing:

- `sample_data.txt`: Simple exponential decay data
- Format: tab-delimited with 'time' and 'concentration' columns

## 🎨 Visualization Options

### Plot Styles

- **Matplotlib/Seaborn**: Static plots with publication-quality styling
- **Plotly**: Interactive plots with zoom, pan, and hover features

### Visualization Types

- **Data vs Model**: Scatter plots with fitted curves
- **Residual Analysis**: Residual plots and Q-Q plots
- **Phase Portraits**: Vector field plots for 2D systems
- **Parameter Distributions**: Histograms with confidence intervals
- **Multi-panel Displays**: Organized subplot layouts

## 🔧 Configuration Options

### Optimization Settings

- **Tolerance**: Convergence tolerance (default: 1e-8)
- **Max Iterations**: Maximum optimization iterations (default: 1000)
- **Multi-start**: Number of random starting points (default: 10)

### Bootstrap Settings

- **Sample Size**: Number of bootstrap samples (10-1000)
- **Method**: Residual resampling or parametric bootstrap
- **Confidence Level**: 90%, 95%, or 99%

### Error Handling

- **Relative Error**: Normalized by data magnitude
- **Absolute Error**: Raw differences
- **Weighted Fitting**: Custom weights for different datasets

## 📈 Performance Considerations

### Optimization Tips

- Use **multi-start optimization** for complex parameter spaces
- Set **realistic parameter bounds** to improve convergence
- Consider **log transformations** for positive-only data
- Use **relative error** for data spanning multiple orders of magnitude

### Bootstrap Analysis

- Start with **fewer samples** (50-100) for initial testing
- Use **residual resampling** for most applications
- **Parametric bootstrap** for well-characterized noise models
- Consider computational time for large datasets

## 📚 Technical Details

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

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Built with:

- [Streamlit](https://streamlit.io/) for the web interface
- [SciPy](https://scipy.org/) for numerical optimization
- [NumPy](https://numpy.org/) for numerical computations
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Matplotlib](https://matplotlib.org/) and [Plotly](https://plotly.com/) for visualization

---

**Happy modeling! 🚀**

For questions or support, please open an issue in the repository.
