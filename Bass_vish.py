import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# Set wide mode 
st.set_page_config(layout="wide")

def bass_diffusion(N, t, p, q, m):
    """Bass Diffusion Model Equation"""
    dNdt = (p + q * (N / m)) * (m - N)
    return dNdt

def solve_bass_model(p, q, m, T=50):
    """Solves the Bass model using numerical integration"""
    t = np.linspace(0, T, 100)
    N0 = 0  # Initial condition: no adopters at t=0
    N = odeint(bass_diffusion, N0, t, args=(p, q, m))
    return t, N.flatten()

def bass_model(t, p, q, M):
    A = (p + q) / M
    B = p * q / M
    return M * A * (1 - np.exp(-A * t)) / (1 + (B / A) * (np.exp(-A * t)))

def generate_random_cumulative_adopters():
    base_values = np.array([0, 2, 7, 13, 19, 25, 30, 34, 38, 41, 43])
    noise = np.random.randint(-2, 2, size=base_values.shape)  # Add small random noise
    return np.clip(base_values + noise, 0, None)  # Ensure values are non-negative

# Initialize session state for analogue data if not already set
if "analogue_data" not in st.session_state:
    st.session_state.analogue_data = [generate_random_cumulative_adopters() for _ in range(5)]

st.title("Bass Diffusion Model with Analogue Curves")
st.sidebar.header("Model Parameters")

# User Inputs
p = st.sidebar.slider("Innovation Coefficient (p)", 0.01, 1.0, 0.03, 0.001)
q = st.sidebar.slider("Imitation Coefficient (q)", 0.01, 1.0, 0.38, 0.01)
m = st.sidebar.slider("Market Size (m)", 1, 1000, 100000, 1000)
T = st.sidebar.slider("Time Period (Years)", 10, 100, 50, 5)
num_curves = st.sidebar.slider("Number of Analogue Curves", 1, 5, 3)

# Solve Main Model
t, N = solve_bass_model(p, q, m, T)

# Create Plotly Figure
fig = go.Figure()

# Main Model Curve
fig.add_trace(go.Scatter(
    x=t, y=N, mode='lines', name=f"Equation Driven Curve (p={p}, q={q})",
    line=dict(color='red'),
    hovertemplate="Time: %{x}<br>Adopters: %{y}<br><b>p:</b> " + str(p) + "<br><b>q:</b> " + str(q)
))

# Generate and plot analogue curves
analogue_results = []
for i in range(num_curves):
    st.sidebar.subheader(f"Analogue {i+1} Settings")
    
    use_synthetic = st.sidebar.checkbox(f"Use Synthetic Data for Analogue {i+1}", value=True, key=f"synthetic_{i}")
    
    if not use_synthetic:
        user_input = st.sidebar.text_input(
            f"Enter comma-separated values for Analogue {i+1}", 
            key=f"user_values_{i}"
        )
        if user_input:
            try:
                user_values = np.array([float(x.strip()) for x in user_input.split(",")])
                st.session_state.analogue_data[i] = user_values
            except ValueError:
                st.sidebar.error(f"Invalid input for Analogue {i+1}. Please enter numbers separated by commas.")

    time_data = np.arange(len(st.session_state.analogue_data[i]))
    adopters_data = st.session_state.analogue_data[i] * 0.001
    
    try:
        params, _ = curve_fit(
            bass_model, 
            time_data, 
            adopters_data, 
            p0=[0.01, 0.3, 100],  # Initial guess
            bounds=([0.001, 0.01, 50], [1.0, 1.0, 100000]),  # Keep p and q in range
            maxfev=5000  # Allow more iterations for convergence
        )
        
        p_analogue, q_analogue, m_analogue = params
        t_analogue, N_analogue = solve_bass_model(p_analogue, q_analogue, m, T)

        fig.add_trace(go.Scatter(
            x=t_analogue, y=N_analogue, mode='lines', 
            name=f"Analogue {i+1} (p={p_analogue:.2f}, q={q_analogue:.2f})",
            line=dict(dash='dash'),
            hovertemplate=f"Time: %{{x}}<br>Adopters: %{{y}}<br><b>p:</b> {p_analogue:.2f}<br><b>q:</b> {q_analogue:.2f}"
        ))
        
        analogue_results.append([f"Analogue {i+1}", np.argmax(np.diff(N_analogue)), p_analogue, q_analogue, m_analogue, int(N_analogue[-1])])
    except:
        st.sidebar.error(f"Curve fitting failed for Analogue {i+1}. Try different values.")

fig.update_layout(
    title="Bass Diffusion Model vs. Analogue Curves",
    xaxis_title="Time (Years)",
    yaxis_title="Cumulative Adopters",
    template="plotly_white",
    hovermode="x unified",
    width=1500,
    height=600
)

# Display Plot
st.plotly_chart(fig, use_container_width=False)

# Display Final Table
st.write("### Summary Table")
table_data = [["Product X", np.argmax(np.diff(N)), p, q, m, int(N[-1])]] + analogue_results
st.table([("Product", "Time TO Peak", "P", "Q", "M", "Adopters")] + table_data)

if st.sidebar.button("Reset Analogues"):
    st.session_state.analogue_data = [generate_random_cumulative_adopters() for _ in range(5)]
    st.rerun()