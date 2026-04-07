import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# -------------------------------
# DATA GENERATION
# -------------------------------
def generate_data(n=100, noise=0.0, outliers=False):
    X = np.linspace(0, 10, n)
    y = 2 * X + 3 + np.random.randn(n) * noise

    if outliers:
        y[:5] += np.random.randn(5) * 20  # add extreme values

    return X, y

# -------------------------------
# MSE FUNCTION
# -------------------------------
def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# -------------------------------
# GRADIENT DESCENT
# -------------------------------
def gradient_descent(X, y, lr=0.01, epochs=50):
    m, b = 0, 0
    history = []

    n = len(X)

    for _ in range(epochs):
        y_pred = m * X + b

        dm = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)

        m -= lr * dm
        b -= lr * db

        loss = compute_mse(y, y_pred)
        history.append((m, b, loss))

    return history

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.title("Controls")

noise = st.sidebar.slider("Noise", 0.0, 10.0, 1.0)
m = st.sidebar.slider("Slope (m)", -5.0, 5.0, 1.0)
b = st.sidebar.slider("Intercept (b)", -10.0, 10.0, 0.0)
lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
epochs = st.sidebar.slider("Iterations", 10, 200, 50)
outliers = st.sidebar.checkbox("Add Outliers")

# -------------------------------
# DATA
# -------------------------------
X, y = generate_data(noise=noise, outliers=outliers)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Line Fit",
    "Error (MSE)",
    "Gradient Descent",
    "Learning Rate"
])

# -------------------------------
# TAB 1: LINE FIT
# -------------------------------
with tab1:
    st.subheader("Data & Line Fit")

    y_pred = m * X + b

    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Data")
    ax.plot(X, y_pred, color='red', label="Model")
    ax.legend()

    st.pyplot(fig)

    st.write("👉 Adjust slope and intercept to see how the line fits the data.")

# -------------------------------
# TAB 2: ERROR VISUALIZATION
# -------------------------------
with tab2:
    st.subheader("Error Visualization (MSE)")

    y_pred = m * X + b
    mse = compute_mse(y, y_pred)

    fig, ax = plt.subplots()
    ax.scatter(X, y)

    for i in range(len(X)):
        ax.plot([X[i], X[i]], [y[i], y_pred[i]], 'r--')

    ax.plot(X, y_pred)
    st.pyplot(fig)

    st.metric("MSE", round(mse, 3))

    st.write("👉 Vertical lines show residuals. Larger gaps = higher error.")

# -------------------------------
# TAB 3: GRADIENT DESCENT
# -------------------------------
with tab3:
    st.subheader("Gradient Descent Optimization")

    history = gradient_descent(X, y, lr, epochs)

    losses = [h[2] for h in history]

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title("Loss vs Iterations")

    st.pyplot(fig)

    st.write("👉 Observe how error decreases over time.")

# -------------------------------
# TAB 4: LEARNING RATE
# -------------------------------
with tab4:
    st.subheader("Learning Rate Comparison")

    lrs = [0.001, 0.01, 0.1]

    fig, ax = plt.subplots()

    for lr_val in lrs:
        history = gradient_descent(X, y, lr_val, epochs)
        losses = [h[2] for h in history]
        ax.plot(losses, label=f"lr={lr_val}")

    ax.legend()
    st.pyplot(fig)

    st.write("👉 Small LR = slow, large LR = unstable, optimal LR = smooth convergence.")