import streamlit as st
import pandas as pd
import pickle

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Smart Grid",
    # page_icon="",
    layout="wide",
)

st.markdown(
    """ 
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2338 0%, #102a42 100%);
        color: #fff;
    }
    .stButton>button {
        background-color: #0f7bff;
        color: white;
        border-radius: 10px;
        height: 3rem;
    }
    .stButton>button:hover {
        background-color: #0c6bed;
    }
    .status-box {
        border-radius: 18px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .stable-bg {
        background-color: #e9f7ef;
        border: 1px solid #4caf50;
    }
    .unstable-bg {
        background-color: #fdecea;
        border: 1px solid #f44336;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# LOAD MODEL
# ==========================================
model = pickle.load(open("best_model.pkl", "rb"))

feature_names = [
    'tau1', 'tau2', 'tau3', 'tau4',
    'p1', 'p2', 'p3', 'p4',
    'g1', 'g2', 'g3', 'g4'
]

feature_labels = {
    "tau1": "⚡ Node 1 Response Speed",
    "tau2": "⚡ Node 2 Response Speed",
    "tau3": "⚡ Node 3 Response Speed",
    "tau4": "⚡ Node 4 Response Speed",

    "p1": "🔋 Node 1 Power Flow",
    "p2": "🔋 Node 2 Power Flow",
    "p3": "🔋 Node 3 Power Flow",
    "p4": "🔋 Node 4 Power Flow",

    "g1": "💰 Node 1 Demand Flexibility",
    "g2": "💰 Node 2 Demand Flexibility",
    "g3": "💰 Node 3 Demand Flexibility",
    "g4": "💰 Node 4 Demand Flexibility"
}

# ==========================================
# HEADER
# ==========================================
st.title(" Smart Grid Stability Monitoring System")
st.markdown(
    """
### Real-Time Grid Stability Assessment Dashboard

Use the sliders to simulate grid conditions and visualize stability risk instantly.
The dashboard is powered by a trained XGBoost model and designed for clarity.
"""
)

st.markdown(
    """
<div style="background: linear-gradient(90deg, rgba(15,123,255,0.12), rgba(15,123,255,0)); padding: 1rem; border-radius: 16px;">
    <strong>Interactive Monitoring</strong> • Predict stability with live parameter updates • Review confidence, risk level, and feature impact.
</div>
""",
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# SIDEBAR INPUTS
# ==========================================
st.sidebar.title("🎛 Control Panel")
st.sidebar.caption("Configure the four smart grid participants.")

with st.sidebar.expander("🔵 Node 1", expanded=True):
    tau1 = st.slider(
        "⚡ Response Speed",
        0.0,
        10.0,
        5.0,
        key="tau1",
        help="Lower values indicate faster reaction to disturbances."
    )
    p1 = st.slider(
        "🔋 Power Flow",
        -5.0,
        5.0,
        0.0,
        key="p1",
        help="Positive values mean this node is supplying electricity. Negative values mean it is consuming electricity."
    )
    g1 = st.slider(
        "💰 Demand Flexibility",
        0.0,
        1.0,
        0.5,
        key="g1",
        help="Higher values mean consumers at this node adjust their electricity usage more when prices change."
    )

with st.sidebar.expander("🟢 Node 2", expanded=True):
    tau2 = st.slider(
        "⚡ Response Speed",
        0.0,
        10.0,
        5.0,
        key="tau2",
        help="Lower values indicate faster reaction to disturbances."
    )
    p2 = st.slider(
        "🔋 Power Flow",
        -5.0,
        5.0,
        0.0,
        key="p2",
        help="Positive values mean this node is supplying electricity. Negative values mean it is consuming electricity."
    )
    g2 = st.slider(
        "💰 Demand Flexibility",
        0.0,
        1.0,
        0.5,
        key="g2",
        help="Higher values mean consumers at this node adjust their electricity usage more when prices change."
    )

with st.sidebar.expander("🟡 Node 3", expanded=True):
    tau3 = st.slider(
        "⚡ Response Speed",
        0.0,
        10.0,
        5.0,
        key="tau3",
        help="Lower values indicate faster reaction to disturbances."
    )
    p3 = st.slider(
        "🔋 Power Flow",
        -5.0,
        5.0,
        0.0,
        key="p3",
        help="Positive values mean this node is supplying electricity. Negative values mean it is consuming electricity."
    )
    g3 = st.slider(
        "💰 Demand Flexibility",
        0.0,
        1.0,
        0.5,
        key="g3",
        help="Higher values mean consumers at this node adjust their electricity usage more when prices change."
    )

with st.sidebar.expander("🔴 Node 4", expanded=True):
    tau4 = st.slider(
        "⚡ Response Speed",
        0.0,
        10.0,
        5.0,
        key="tau4",
        help="Lower values indicate faster reaction to disturbances."
    )
    p4 = st.slider(
        "🔋 Power Flow",
        -5.0,
        5.0,
        0.0,
        key="p4",
        help="Positive values mean this node is supplying electricity. Negative values mean it is consuming electricity."
    )
    g4 = st.slider(
        "💰 Demand Flexibility",
        0.0,
        1.0,
        0.5,
        key="g4",
        help="Higher values mean consumers at this node adjust their electricity usage more when prices change."
    )

st.sidebar.info(
    "Adjust each node's behaviour and click 'Predict Stability' to assess whether the smart grid remains stable."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
### Quick Tips
- Keep Response Speed values moderate for responsive control.
- Balance Power Flow values to avoid overload.
- Use Demand Flexibility values that reflect realistic customer behavior.

Explore multiple scenarios to see how the predicted status changes.
"""
)

# ==========================================
# PREDICTION
# ==========================================
if st.sidebar.button("Predict Stability"):

    X = pd.DataFrame([[
        tau1, tau2, tau3, tau4,
        p1, p2, p3, p4,
        g1, g2, g3, g4
    ]], columns=feature_names)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    # ==========================================
    # STATUS
    # ==========================================
    if prediction == 1:
        status = "🟢 Stable"
        confidence = probabilities[1] * 100
        status_class = "stable-bg"
    else:
        status = "🔴 Unstable"
        confidence = probabilities[0] * 100
        status_class = "unstable-bg"

    # Risk Level
    if confidence >= 90:
        risk_level = "Very High"
    elif confidence >= 75:
        risk_level = "High"
    elif confidence >= 50:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # ==========================================
    # METRIC CARDS
    # ==========================================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Grid Status", status)
    col2.metric("Confidence", f"{confidence:.2f}%")
    col3.metric("Risk Level", risk_level)
    col4.metric("Model", "XGBoost")

    st.progress(min(int(confidence), 100))
    st.divider()

    # ==========================================
    # RESULT BOX + PARAMETER CHARTS
    # ==========================================
    left_col, right_col = st.columns([2, 1], gap="large")

    with left_col:
        st.markdown(f"<div class='status-box {status_class}'>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(
                f"""
### ✅ GRID STABLE

**Confidence:** {confidence:.2f}%

#### Operational Guidance
• No immediate corrective action required.
• Grid operating within acceptable limits.
• Continue routine monitoring.
• Maintain standard operating procedures.
"""
            )
        else:
            st.markdown(
                f"""
### ⚠️ GRID INSTABILITY DETECTED

**Risk Probability:** {confidence:.2f}%

#### Recommended Actions
• Reduce sudden load fluctuations.
• Monitor generator response times.
• Investigate delayed reaction nodes.
• Verify power balancing mechanisms.
• Increase operational surveillance.
"""
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.subheader("📊 Current Parameter Profile")
        param_df = pd.DataFrame({
            "Value": [
                tau1, tau2, tau3, tau4,
                p1, p2, p3, p4,
                g1, g2, g3, g4
            ]
        }, index=[feature_labels[f] for f in feature_names])
        st.bar_chart(param_df)

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================
    st.subheader("📊 Top Factors Affecting Stability")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(5)

    importance_df["Feature"] = importance_df["Feature"].map(feature_labels)
    importance_df["Importance (%)"] = (
        importance_df["Importance"] * 100
    ).round(2)

    info_col, chart_col = st.columns([1, 1], gap="large")

    with info_col:
        st.dataframe(
            importance_df[["Feature", "Importance (%)"]],
            use_container_width=True
        )

    with chart_col:
        st.subheader("📈 Feature Importance")
        chart_df = importance_df.set_index("Feature")
        st.bar_chart(chart_df["Importance"])

    # ==========================================
    # INPUT SUMMARY
    # ==========================================
    st.subheader("📋 Current Grid Parameters")

    display_X = X.rename(columns=feature_labels)
    st.dataframe(
        display_X,
        use_container_width=True
    )

    st.markdown(
        """
---
<div style='display:flex; justify-content:space-between; align-items:center; color:#555;'>
  <span>Model: <strong>XGBoost</strong> | Accuracy: <strong>97%</strong> | Dataset Size: <strong>60,000</strong></span>
  <span>Developed by Siddhant Gupta</span>
</div>
""",
        unsafe_allow_html=True
    )

# ==========================================
# PARAMETER GUIDE
# ==========================================
st.divider()

st.header("📖 Parameter Guide")

with st.expander("⚡ Response Speed"):
    st.markdown(
        """
- Indicates how quickly a node reacts to sudden changes in the grid.
- Lower values mean faster response.
- Higher values mean slower response.
"""
    )

with st.expander("🔋 Power Flow"):
    st.markdown(
        """
- Positive values indicate electricity generation.
- Negative values indicate electricity consumption.
- Values close to zero indicate balanced operation.
"""
    )

with st.expander("💰 Demand Flexibility"):
    st.markdown(
        """
- Indicates how willing consumers are to adjust electricity usage when prices change.
- Higher values mean greater responsiveness.
- Lower values mean little change in behavior.
"""
    )

# ==========================================
# FOOTER
# ==========================================
st.divider()

st.caption(
    """
Model: XGBoost | Accuracy: 97% | Dataset Size: 60,000 Samples

Developed by Siddhant Gupta
"""
)