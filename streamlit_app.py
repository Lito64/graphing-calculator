import streamlit as st
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
import plotly.graph_objects as go

# --- Page setup ---
st.set_page_config(page_title="Graphing Calculator", page_icon="📈", layout="wide")
st.title("📈 Graphing Calculator")
st.write(
    "Type a function of **x** and I'll plot it. "
    "Optionally overlay the first and second derivatives."
)

# --- Symbol + parser setup ---
# 'x' is our one allowed variable. The transformations let people type
# math in a natural way: 3x instead of 3*x, and x^2 instead of x**2.
x = sp.symbols("x")
transformations = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# --- Sidebar controls ---
with st.sidebar:
    st.header("Settings")
    func_input = st.text_input("Function f(x)", value="x**2 + 3*x")
    show_first = st.checkbox("Show first derivative  f′(x)")
    show_second = st.checkbox("Show second derivative  f″(x)")

    st.subheader("X-axis range")
    x_min = st.number_input("x min", value=-10.0)
    x_max = st.number_input("x max", value=10.0)
    num_points = st.slider(
        "Resolution (points)", min_value=100, max_value=2000, value=500, step=100
    )

# --- Validate the range ---
if x_min >= x_max:
    st.error("x min must be less than x max.")
    st.stop()

# --- Parse the typed function into a symbolic expression ---
try:
    expr = parse_expr(func_input, transformations=transformations, local_dict={"x": x})
except Exception as e:
    st.error(f"Couldn't understand that function. Error: {e}")
    st.stop()

# Only 'x' is allowed as a variable. Catch typos like a stray 'y'.
extra_symbols = expr.free_symbols - {x}
if extra_symbols:
    names = ", ".join(str(s) for s in extra_symbols)
    st.error(f"Only 'x' is allowed as a variable, but I also found: {names}")
    st.stop()

# --- Let SymPy do the calculus for us ---
first_derivative = sp.diff(expr, x)
second_derivative = sp.diff(expr, x, 2)

# --- Show the symbolic forms nicely with LaTeX ---
st.latex(f"f(x) = {sp.latex(expr)}")
if show_first:
    st.latex(f"f'(x) = {sp.latex(first_derivative)}")
if show_second:
    st.latex(f"f''(x) = {sp.latex(second_derivative)}")

# --- Convert symbolic expressions into fast numeric functions ---
f = sp.lambdify(x, expr, modules=["numpy"])
f1 = sp.lambdify(x, first_derivative, modules=["numpy"])
f2 = sp.lambdify(x, second_derivative, modules=["numpy"])

# --- Evaluate over the chosen range ---
xs = np.linspace(x_min, x_max, num_points)


def safe_eval(func, xs):
    """Evaluate a lambdified function, tolerating constants and bad domains."""
    with np.errstate(all="ignore"):  # silence divide-by-zero / log of negatives
        ys = func(xs)
    ys = np.asarray(ys, dtype=float)
    if ys.shape == ():  # a constant (e.g. derivative of x**2 is just 2)
        ys = np.full_like(xs, float(ys))
    return ys


# --- Build the interactive plot ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=safe_eval(f, xs), mode="lines", name="f(x)"))

if show_first:
    fig.add_trace(go.Scatter(x=xs, y=safe_eval(f1, xs), mode="lines", name="f′(x)"))
if show_second:
    fig.add_trace(go.Scatter(x=xs, y=safe_eval(f2, xs), mode="lines", name="f″(x)"))

fig.add_hline(y=0, line_width=1, line_color="gray")
fig.add_vline(x=0, line_width=1, line_color="gray")
fig.update_layout(
    xaxis_title="x",
    yaxis_title="y",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig, use_container_width=True)

# --- Quick help ---
with st.expander("How to type functions"):
    st.markdown(
        """
        - Use **`x`** as the variable.
        - Powers: `x**2` or `x^2` both work.
        - Multiplication can be implicit: `3x` works, so does `3*x`.
        - Built-in functions: `sin(x)`, `cos(x)`, `tan(x)`, `exp(x)`, `log(x)`, `sqrt(x)`, and more.
        - Constants: `pi`, `E`.
        - Examples: `sin(x)`, `x**3 - 2*x`, `exp(-x**2)`, `log(x)`
        """
    )
