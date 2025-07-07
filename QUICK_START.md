# 🚀 Quick Start Guide

## Getting Started in 5 Minutes

### 1. Install and Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### 2. Test with Sample Data

#### Option A: Simple Single Dataset
1. Go to **📁 Data Upload** tab
2. Upload `sample_data.txt` with dataset name "concentration"
3. Go to **🧬 ODE Definition** tab
4. Check "Use an example ODE system" and select "Exponential Decay"
5. Click "Load Example"
6. Set initial conditions: y[0](0) = 10.0
7. Go to **📊 Model Fitting** tab
8. Click "🚀 Run Advanced Model Fitting"

#### Option B: Multi-Dataset Example (Viral Dynamics)
1. Go to **📁 Data Upload** tab
2. Upload `viral_load_sample.txt` with dataset name "viral_load"
3. Upload `interferon_sample.txt` with dataset name "interferon"
4. Go to **🧬 ODE Definition** tab
5. Enter this ODE system:
```python
T, R, I, V, F = y
dTdt = -beta * T * V - gamma * T * F
dRdt = gamma * T * F - rho * R
dIdt = beta * T * V - delta * I
dVdt = p * I - c * V
dFdt = I - alpha * F
return [dTdt, dRdt, dIdt, dVdt, dFdt]
```
6. Set number of state variables: 5
7. Set initial conditions: [1.0, 0.0, 0.0, 4.2, 0.1]
8. Map datasets:
   - viral_load → y[3] (V)
   - interferon → y[4] (F)
9. Go to **📊 Model Fitting** tab
10. Set parameter bounds (suggested):
    - beta: [1e-6, 10.0]
    - gamma: [1e-6, 10.0]
    - rho: [1e-6, 10.0]
    - delta: [1e-6, 10.0]
    - p: [0.1, 1000.0]
    - c: [1e-6, 10.0]
    - alpha: [1e-6, 100.0]
11. Click "🚀 Run Advanced Model Fitting"

### 3. Explore Results
- Check **📈 Results** tab for fitted parameters and visualizations
- Try **🎯 Bootstrap Analysis** for uncertainty quantification
- Export results packages for further analysis

### 4. Advanced Features to Try

#### Multi-start Optimization
- In sidebar, enable "Multi-start Optimization"
- Set number of starts to 5-10 for robust fitting

#### Bootstrap Analysis
- Go to **🎯 Bootstrap Analysis** tab
- Set 50-100 bootstrap samples (start small)
- Choose "Residual Resampling"
- Run analysis to get parameter confidence intervals

#### Custom Visualization
- Try both "seaborn" and "plotly" plot styles
- Enable "Show Phase Portrait" for 2D systems
- Check "Show Parameter Distributions" for bootstrap results

## 🎯 Your Original Workflow

To replicate your H3N2 viral dynamics workflow:

1. **Upload your data files** with 'time' and 'value' columns
2. **Define your Model function** in the ODE Definition tab
3. **Set parameter bounds** matching your original bounds
4. **Enable multi-start optimization** for robustness
5. **Run bootstrap analysis** with 100+ samples
6. **Export results** for further analysis

The tool now supports:
- ✅ Multiple datasets (viral load + interferon)
- ✅ Complex ODE systems (5+ variables)
- ✅ Bootstrap uncertainty analysis
- ✅ Comprehensive result export
- ✅ Advanced optimization options
- ✅ Interactive visualizations

## 🔧 Troubleshooting

### Common Issues
- **"No module named 'ode_examples'"**: Make sure all files are in the same directory
- **Data upload errors**: Check that your files have 'time' and 'value' columns
- **Optimization fails**: Try wider parameter bounds or different optimization methods
- **Bootstrap takes too long**: Reduce number of samples or use simpler models

### Performance Tips
- Start with fewer bootstrap samples (50-100) for testing
- Use relative error for data spanning multiple orders of magnitude
- Try different optimization methods if one fails
- Set realistic parameter bounds to improve convergence

## 📱 Need Help?

- Check the **📚 Examples** tab for built-in ODE systems
- Refer to the full README.md for detailed documentation
- The tool includes helpful tooltips and error messages
- Try the sample data first to understand the workflow

**Happy modeling! 🚀** 