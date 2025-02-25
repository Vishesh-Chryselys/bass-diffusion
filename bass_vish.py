import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint


# set wide mode 
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

# Streamlit UI
st.title("Bass Diffusion Model with Analogue Curves")
st.sidebar.header("Model Parameters")

# User Inputs
p = st.sidebar.slider("Innovation Coefficient (p)", 0.00, 1.0, 0.03, 0.001)
q = st.sidebar.slider("Imitation Coefficient (q)", 0.00, 1.0, 0.38, 0.01)
m = st.sidebar.slider("Market Size (m)", 1000, 1000000, 100000, 1000)
T = st.sidebar.slider("Time Period (Years)", 10, 100, 50, 5)

# Analogue Curves (predefined variations)
p_analogue = [0.2, 0.9, 0.6]  
q_analogue = [0.5, 0.7, 0.3]  

# Solve Models
t, N = solve_bass_model(p, q, m, T)
t1, N1 = solve_bass_model(p_analogue[0], q_analogue[0], m, T)
t2, N2 = solve_bass_model(p_analogue[1], q_analogue[1], m, T)
t3, N3 = solve_bass_model(p_analogue[2], q_analogue[2], m, T)


# Create Plotly Figure
fig = go.Figure()

# Main Model Curve
fig.add_trace(go.Scatter(
    x=t, y=N, mode='lines', name=f"Equation Driven Curve (p={p}, q={q})",
    line=dict(color='red'),
    hovertemplate="Time: %{x}<br>Adopters: %{y}<br><b>p:</b> " + str(p) + "<br><b>q:</b> " + str(q)
))

# Analogue Curves
fig.add_trace(go.Scatter(
    x=t1, y=N1, mode='lines', name=f"Analogue 1 (p={p_analogue[0]}, q={q_analogue[0]})",
    line=dict(color='green', dash='dash'),
    hovertemplate="Time: %{x}<br>Adopters: %{y}<br><b>p:</b> " + str(p_analogue[0]) + "<br><b>q:</b> " + str(q_analogue[0])
))

fig.add_trace(go.Scatter(
    x=t2, y=N2, mode='lines', name=f"Analogue 2 (p={p_analogue[1]}, q={q_analogue[1]})",
    line=dict(color='blue', dash='dot'),
    hovertemplate="Time: %{x}<br>Adopters: %{y}<br><b>p:</b> " + str(p_analogue[1]) + "<br><b>q:</b> " + str(q_analogue[1])
))

fig.add_trace(go.Scatter(
    x=t3, y=N3, mode='lines', name=f"Analogue 3 (p={p_analogue[2]}, q={q_analogue[2]})",
    line=dict(color='yellow', dash='longdash'),
    hovertemplate="Time: %{x}<br>Adopters: %{y}<br><b>p:</b> " + str(p_analogue[2]) + "<br><b>q:</b> " + str(q_analogue[2])
))

fig.update_layout(
    title="Bass Diffusion Model vs. Analogue Curves",
    xaxis_title="Time (Years)",
    yaxis_title="Cumulative Adopters",
    template="plotly_white",  # You can use "plotly_dark" if you prefer
    hovermode="x unified",
    width=1500,  # Increased width
    height=600   # Increased height
)

# Display Plot
st.plotly_chart(fig, use_container_width=False)  # Set to False to respect width & height


# Display Final Adoption Counts
st.write("### Final Number of Adopters:")
st.write(f"- **Equation Driven Curve:** {int(N[-1])}")
st.write(f"- **Analogue Curve 1:** {int(N1[-1])}")
st.write(f"- **Analogue Curve 2:** {int(N2[-1])}")
st.write(f"- **Analogue Curve 3:** {int(N3[-1])}")

peak_index = np.argmax(np.diff(N))
time_to_peak = t[peak_index]
st.write(f"### Time to Peak Adoption: {time_to_peak:.2f} years")



