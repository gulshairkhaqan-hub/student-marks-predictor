import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_URL = (
    "https://raw.githubusercontent.com/gulshairkhaqan-hub/"
    "student-marks-predictor/main/scores.csv"
)

st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    .stApp {
        background: linear-gradient(160deg, #0a0e17 0%, #0d1526 50%, #0a1628 100%);
        color: #e8f4ff;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }

    h1 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2.4rem !important;
        background: linear-gradient(90deg, #00d4ff, #4da6ff, #00ffcc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem !important;
        letter-spacing: -0.02em;
    }

    .subtitle {
        color: #7eb8d4;
        font-size: 1rem;
        margin-bottom: 2rem;
        letter-spacing: 0.04em;
    }

    .section-label {
        color: #9ecae8;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .helper-text {
        color: #5a8aaa;
        font-size: 0.85rem;
        margin-top: -0.5rem;
        margin-bottom: 1.25rem;
    }

    div[data-testid="stNumberInput"] label {
        color: #9ecae8 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #111b2e !important;
        border: 1px solid #1e4a6e !important;
        color: #00d4ff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.1rem !important;
        border-radius: 8px !important;
    }

    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, #0066cc, #00a8cc) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        transition: box-shadow 0.2s ease;
    }

    div[data-testid="stButton"] button:hover {
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4) !important;
    }

    .prediction-box {
        background: linear-gradient(135deg, #0f2847 0%, #122a4a 100%);
        border: 1px solid #00d4ff;
        border-radius: 12px;
        padding: 1.75rem 2rem;
        margin: 1.5rem 0 2rem 0;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.15);
    }

    .prediction-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3rem;
        font-weight: 700;
        color: #00ffcc;
        line-height: 1.2;
    }

    .prediction-label {
        color: #7eb8d4;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .prediction-grade {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #4da6ff;
        margin-top: 0.5rem;
    }

    .stats-container {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 1.5rem;
    }

    .stat-card {
        flex: 1;
        min-width: 140px;
        background: #111b2e;
        border: 1px solid #1e4a6e;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }

    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #00d4ff;
    }

    .stat-label {
        color: #7eb8d4;
        font-size: 0.8rem;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    h3 {
        color: #9ecae8 !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_train():
    df = pd.read_csv(DATA_URL)
    X = df[["Hours"]]
    y = df["Scores"]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return df, model, y_pred


def get_grade(score: float) -> str:
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    return "D"


def plot_dark(df, model):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0a0e17")
    ax.set_facecolor("#0d1526")

    ax.scatter(
        df["Hours"],
        df["Scores"],
        color="#00d4ff",
        s=70,
        alpha=0.85,
        edgecolors="#4da6ff",
        linewidths=0.5,
        label="Actual data",
    )

    hours_line = [df["Hours"].min(), df["Hours"].max()]
    scores_line = model.predict([[hours_line[0]], [hours_line[1]]])
    ax.plot(
        hours_line,
        scores_line,
        color="#00ffcc",
        linewidth=2.5,
        label="Regression line",
    )

    ax.set_xlabel("Hours studied", color="#9ecae8", fontsize=11)
    ax.set_ylabel("Scores", color="#9ecae8", fontsize=11)
    ax.set_title("Study Hours vs Scores", color="#e8f4ff", fontsize=13, pad=12)
    ax.tick_params(colors="#7eb8d4")
    ax.grid(True, alpha=0.2, color="#1e4a6e")
    for spine in ax.spines.values():
        spine.set_color("#1e4a6e")
    ax.legend(facecolor="#111b2e", edgecolor="#1e4a6e", labelcolor="#9ecae8")

    fig.tight_layout()
    return fig


df, model, y_pred = load_and_train()

st.title("Student Marks Predictor")
st.markdown('<p class="subtitle">Powered by Machine Learning</p>', unsafe_allow_html=True)

st.markdown('<p class="section-label">Hours Studied</p>', unsafe_allow_html=True)
hours = st.number_input(
    "Hours studied",
    min_value=0.5,
    max_value=10.0,
    step=0.25,
    value=5.0,
    label_visibility="collapsed",
)
st.markdown(
    '<p class="helper-text">Enter hours between 0.5 and 10</p>',
    unsafe_allow_html=True,
)

if st.button("Predict"):
    st.session_state["predicted_score"] = float(model.predict([[hours]])[0])

if "predicted_score" in st.session_state:
    score = st.session_state["predicted_score"]
    grade = get_grade(score)
    st.markdown(
        f"""
        <div class="prediction-box">
            <div class="prediction-label">Predicted Score</div>
            <div class="prediction-score">{score:.2f}</div>
            <div class="prediction-grade">Grade: {grade}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

fig = plot_dark(df, model)
st.pyplot(fig)
plt.close(fig)

r2 = r2_score(df["Scores"], y_pred)
mae = mean_absolute_error(df["Scores"], y_pred)
rmse = mean_squared_error(df["Scores"], y_pred) ** 0.5

st.markdown("### Model Statistics")
st.markdown(
    f"""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-value">{r2:.4f}</div>
            <div class="stat-label">R² Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{mae:.4f}</div>
            <div class="stat-label">MAE</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{rmse:.4f}</div>
            <div class="stat-label">RMSE</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
