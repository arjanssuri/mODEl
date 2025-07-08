# 🚀 mODEl Enhanced Batch Jobs with Advanced Analytics

This document outlines the comprehensive batch processing capabilities in mODEl that preserve all advanced analytics features.

## ✨ Key Improvements

### 🔧 **UI Stability Fixes**
- **Fixed navigation glitches** when changing state variables or initial conditions
- **Stable widget keys** prevent unexpected page navigation
- **Preserved state across job switches** maintains your work seamlessly
- **Consistent session state management** prevents data loss

### 📊 **Complete Advanced Analytics Preservation**
Every batch job now maintains full access to:

#### 🎯 **Parameter Configuration**
- **Code-based bounds definition** with mathematical expressions
- **Individual parameter bounds** with validation
- **Initial guess preservation** across job switches
- **Template loading** for common parameter configurations

#### ⚙️ **Optimization Settings**
- **Multiple algorithms**: L-BFGS-B, Nelder-Mead, SLSQP, Powell, TNC, Differential Evolution
- **Multi-start optimization** with configurable starts
- **Convergence settings**: Tolerance and max iterations
- **Error metrics**: Relative vs absolute error options

#### 🎲 **Bootstrap Analysis**
- **Residual resampling** and **parametric bootstrap** methods
- **Configurable confidence levels**: 90%, 95%, 99%
- **Real-time progress logging** with customizable frequency
- **Parameter uncertainty quantification** with confidence intervals

#### 📈 **Visualization Settings**
- **Plot style preferences**: Plotly vs Matplotlib
- **Phase portrait analysis** for 2D systems
- **Parameter distribution plots** from bootstrap results
- **Customizable display options**

## 🔄 **Batch Processing Workflow**

### 1. **Upload and Organize**
```
📁 Upload multiple files
├── Choose organization method:
│   ├── 🔸 Separate job per file
│   ├── 🔸 Group by filename prefix  
│   └── 🔸 Group all into one job
└── 📋 Jobs created with full analytics support
```

### 2. **Job Management**
```
📊 Job Summary Panel
├── 🔍 Status tracking for each job
├── ✅ ODE definition status
├── ✅ Parameter fitting status
├── ✅ Bootstrap analysis status
├── ✅ Advanced bounds configuration
└── ✅ Full analytics capability indicator
```

### 3. **Advanced Analytics Workflow**
```
🔄 Per Job Analytics
├── 📁 Load job → All settings preserved
├── 🧬 Define ODE → Auto-detection maintained
├── 🎯 Set initial conditions → Smart options available
├── ⚙️ Configure parameters → Advanced bounds preserved
├── 📊 Run fitting → Optimization settings maintained
├── 🎲 Bootstrap analysis → All settings preserved
├── 💾 Save state → Complete state preservation
└── 🔄 Switch jobs → Seamless transition
```

## 🎯 **Smart Initial Conditions**

### **Three Methods Available**:

1. **Manual Input**
   - Traditional number inputs
   - Values preserved across job switches
   - Stable widget keys prevent glitches

2. **Use First Data Values** 
   - Automatically extracts from mapped datasets
   - Perfect for t=0 measurements
   - Dynamic updating based on mapping

3. **Dataset-Specific First Values**
   - Choose specific dataset per variable
   - Maximum flexibility for complex systems
   - Manual override options available

## 🔧 **Technical Improvements**

### **Session State Management**
```python
# Enhanced state preservation
if 'optimization_settings' not in st.session_state:
    st.session_state.optimization_settings = {
        'method': 'L-BFGS-B',
        'tolerance': 1e-8,
        'max_iter': 1000,
        'multi_start': False,
        'n_starts': 10,
        'use_relative_error': True
    }
```

### **Stable Widget Keys**
```python
# Prevents UI navigation glitches
key=f"manual_ic_{i}_{job_key}"  # Stable across state changes
key=f"dataset_map_{dataset_name}_{job_key}"  # Job-specific keys
```

### **Complete Job State**
```python
# Full analytics state preservation
job_data = {
    'datasets': datasets,
    'ode_system': ode_code,
    'param_names': parameters,
    'initial_conditions': initial_values,
    'bounds_code': bounds_definition,
    'optimization_settings': opt_config,
    'bootstrap_settings': bootstrap_config,
    'visualization_settings': viz_config,
    'fit_results': results,
    'bootstrap_results': uncertainty_analysis
}
```

## 📋 **Feature Verification Checklist**

When working with batch jobs, verify these features are available:

### ✅ **Core Functionality**
- [ ] Multiple file upload and organization
- [ ] Job creation with proper dataset grouping
- [ ] Job loading/saving without data loss
- [ ] Seamless job switching

### ✅ **Advanced Analytics**
- [ ] Code-based parameter bounds definition
- [ ] Multi-start optimization configuration
- [ ] Bootstrap analysis with real-time logging
- [ ] Parameter distribution visualization
- [ ] Complete state preservation across jobs

### ✅ **UI Stability**
- [ ] No navigation glitches when changing initial conditions
- [ ] Preserved values when switching between jobs
- [ ] Stable widget behavior during state variable changes
- [ ] Consistent session state management

### ✅ **Smart Features**
- [ ] Automatic initial condition setting from data
- [ ] Dataset-specific initial condition mapping
- [ ] Parameter bounds validation and analysis
- [ ] Real-time bootstrap progress monitoring

## 🚀 **Usage Examples**

### **Example 1: Viral Dynamics Multi-Dataset Analysis**
```
📁 Upload Files:
├── experiment1_viral_load.txt
├── experiment1_interferon.txt
├── experiment2_viral_load.txt
└── experiment2_interferon.txt

🔄 Organization: Group by prefix
├── Job: experiment1 (viral_load + interferon)
└── Job: experiment2 (viral_load + interferon)

🎯 Per Job:
├── Define viral dynamics ODE system
├── Map datasets: viral_load→V, interferon→F  
├── Set initial conditions from first data values
├── Configure parameter bounds with code
├── Run multi-start optimization
├── Perform bootstrap uncertainty analysis
└── Export complete results package
```

### **Example 2: Parameter Sensitivity Study**
```
📁 Upload Files:
├── condition_A_data.txt
├── condition_B_data.txt
└── condition_C_data.txt

🔄 Organization: Separate job per file
├── Job: condition_A
├── Job: condition_B  
└── Job: condition_C

🎯 Analysis:
├── Same ODE system for all jobs
├── Different initial conditions per job
├── Consistent parameter bounds across jobs
├── Bootstrap analysis for uncertainty
└── Compare parameter estimates across conditions
```

## 💡 **Best Practices**

1. **Job Organization**
   - Use descriptive filenames for auto-grouping
   - Group related datasets into single jobs
   - Save job state frequently during analysis

2. **Advanced Analytics**
   - Define parameter bounds carefully for each system
   - Use bootstrap analysis for parameter uncertainty
   - Export complete results packages for reproducibility

3. **UI Navigation**
   - Use job loading/saving to preserve complex configurations
   - Leverage smart initial conditions for efficiency
   - Take advantage of real-time bootstrap monitoring

---

**🔬 mODEl by Dobrovolny Laboratory, Texas Christian University**

*For support: Check README.md troubleshooting section or the main documentation* 