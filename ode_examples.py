"""
Common ODE examples and helper functions for the ODE Model Fitting Tool
"""

import numpy as np
from typing import Dict, List, Tuple

# Common ODE System Examples
ODE_EXAMPLES = {
    "Exponential Decay": {
        "description": "Simple exponential decay model: dy/dt = -k*y",
        "code": """dydt = [-k * y[0]]
return dydt""",
        "parameters": ["k"],
        "initial_guess": {"k": 0.5},
        "bounds": {"k": (0.0, 10.0)}
    },
    
    "Logistic Growth": {
        "description": "Logistic growth model: dy/dt = r*y*(1 - y/K)",
        "code": """dydt = [r * y[0] * (1 - y[0]/K)]
return dydt""",
        "parameters": ["r", "K"],
        "initial_guess": {"r": 1.0, "K": 100.0},
        "bounds": {"r": (0.0, 10.0), "K": (0.0, 1000.0)}
    },
    
    "SIR Model": {
        "description": "Epidemiological SIR model",
        "code": """# S: Susceptible, I: Infected, R: Recovered
N = y[0] + y[1] + y[2]  # Total population
dS = -beta * y[0] * y[1] / N
dI = beta * y[0] * y[1] / N - gamma * y[1]
dR = gamma * y[1]
dydt = [dS, dI, dR]
return dydt""",
        "parameters": ["beta", "gamma"],
        "initial_guess": {"beta": 0.5, "gamma": 0.1},
        "bounds": {"beta": (0.0, 2.0), "gamma": (0.0, 1.0)}
    },
    
    "Lotka-Volterra": {
        "description": "Predator-prey dynamics",
        "code": """# y[0]: Prey, y[1]: Predator
dydt = [
    alpha * y[0] - beta * y[0] * y[1],
    delta * y[0] * y[1] - gamma * y[1]
]
return dydt""",
        "parameters": ["alpha", "beta", "delta", "gamma"],
        "initial_guess": {"alpha": 1.0, "beta": 0.1, "delta": 0.075, "gamma": 1.5},
        "bounds": {
            "alpha": (0.0, 5.0), 
            "beta": (0.0, 1.0), 
            "delta": (0.0, 1.0), 
            "gamma": (0.0, 5.0)
        }
    },
    
    "Chemical Reaction A->B->C": {
        "description": "Sequential first-order reactions",
        "code": """# y[0]: [A], y[1]: [B], y[2]: [C]
dydt = [
    -k1 * y[0],
    k1 * y[0] - k2 * y[1],
    k2 * y[1]
]
return dydt""",
        "parameters": ["k1", "k2"],
        "initial_guess": {"k1": 1.0, "k2": 0.5},
        "bounds": {"k1": (0.0, 10.0), "k2": (0.0, 10.0)}
    },
    
    "Enzyme Kinetics": {
        "description": "Michaelis-Menten enzyme kinetics",
        "code": """# y[0]: Substrate concentration
dydt = [-Vmax * y[0] / (Km + y[0])]
return dydt""",
        "parameters": ["Vmax", "Km"],
        "initial_guess": {"Vmax": 10.0, "Km": 5.0},
        "bounds": {"Vmax": (0.0, 100.0), "Km": (0.0, 100.0)}
    },
    
    "Damped Oscillator": {
        "description": "Damped harmonic oscillator: x'' + 2*zeta*omega*x' + omega^2*x = 0",
        "code": """# y[0]: position, y[1]: velocity
dydt = [
    y[1],
    -2 * zeta * omega * y[1] - omega**2 * y[0]
]
return dydt""",
        "parameters": ["omega", "zeta"],
        "initial_guess": {"omega": 1.0, "zeta": 0.1},
        "bounds": {"omega": (0.1, 10.0), "zeta": (0.0, 2.0)}
    },
    
    "Van der Pol Oscillator": {
        "description": "Non-linear oscillator with limit cycle",
        "code": """# y[0]: position, y[1]: velocity
dydt = [
    y[1],
    mu * (1 - y[0]**2) * y[1] - y[0]
]
return dydt""",
        "parameters": ["mu"],
        "initial_guess": {"mu": 1.0},
        "bounds": {"mu": (0.0, 5.0)}
    }
}


def generate_synthetic_data(ode_func, t_span: Tuple[float, float], 
                          initial_conditions: List[float], 
                          parameters: Dict[str, float],
                          n_points: int = 50,
                          noise_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for testing ODE fitting.
    
    Parameters:
    -----------
    ode_func : callable
        ODE function that takes (y, t, *params)
    t_span : tuple
        (t_start, t_end) for time range
    initial_conditions : list
        Initial values for state variables
    parameters : dict
        Parameter names and values
    n_points : int
        Number of data points to generate
    noise_level : float
        Relative noise level (0.05 = 5% noise)
    
    Returns:
    --------
    t_data : np.ndarray
        Time points
    y_data : np.ndarray
        Noisy "experimental" data
    """
    from scipy.integrate import odeint
    
    # Generate time points
    t_data = np.linspace(t_span[0], t_span[1], n_points)
    
    # Solve ODE
    param_values = list(parameters.values())
    y_true = odeint(ode_func, initial_conditions, t_data, args=tuple(param_values))
    
    # Add noise
    noise = np.random.normal(0, noise_level, y_true.shape)
    y_data = y_true * (1 + noise)
    
    return t_data, y_data


def calculate_sensitivity_matrix(ode_func, t_data: np.ndarray, 
                               initial_conditions: List[float],
                               parameters: Dict[str, float],
                               delta: float = 1e-5) -> np.ndarray:
    """
    Calculate sensitivity matrix for parameter estimation.
    
    Sensitivity matrix S_ij = dy_i/dp_j
    """
    from scipy.integrate import odeint
    
    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    n_params = len(param_values)
    
    # Baseline solution
    y_base = odeint(ode_func, initial_conditions, t_data, args=tuple(param_values))
    
    # Initialize sensitivity matrix
    sensitivity = np.zeros((y_base.size, n_params))
    
    # Calculate sensitivities
    for j, param_val in enumerate(param_values):
        # Perturb parameter
        params_perturbed = param_values.copy()
        params_perturbed[j] = param_val * (1 + delta)
        
        # Solve with perturbed parameter
        y_perturbed = odeint(ode_func, initial_conditions, t_data, 
                           args=tuple(params_perturbed))
        
        # Calculate sensitivity
        sensitivity[:, j] = ((y_perturbed - y_base) / (param_val * delta)).flatten()
    
    return sensitivity


def estimate_parameter_confidence(sensitivity_matrix: np.ndarray, 
                                residuals: np.ndarray,
                                alpha: float = 0.05) -> np.ndarray:
    """
    Estimate parameter confidence intervals using linear approximation.
    
    Returns standard errors for each parameter.
    """
    from scipy import stats
    
    # Calculate covariance matrix
    n_data = len(residuals)
    n_params = sensitivity_matrix.shape[1]
    
    # Residual sum of squares
    rss = np.sum(residuals**2)
    
    # Estimate of variance
    sigma2 = rss / (n_data - n_params)
    
    # Covariance matrix
    try:
        cov_matrix = sigma2 * np.linalg.inv(sensitivity_matrix.T @ sensitivity_matrix)
        std_errors = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        # If matrix is singular, return NaN
        std_errors = np.full(n_params, np.nan)
    
    return std_errors


def create_phase_portrait(ode_func, param_values: List[float],
                         x_range: Tuple[float, float], 
                         y_range: Tuple[float, float],
                         n_grid: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create phase portrait data for 2D ODE systems.
    """
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(x, y)
    
    # Calculate derivatives at each point
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(n_grid):
        for j in range(n_grid):
            derivatives = ode_func([X[i,j], Y[i,j]], 0, *param_values)
            U[i,j] = derivatives[0]
            V[i,j] = derivatives[1]
    
    # Normalize arrows
    N = np.sqrt(U**2 + V**2)
    U_norm = U / (N + 1e-10)
    V_norm = V / (N + 1e-10)
    
    return X, Y, U_norm, V_norm 