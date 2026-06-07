# =============================================================================
#  HOUSE PRICE PREDICTION — STREAMLIT APP
#  Internship Project — Synent Technology | Task T-8
#  Run: streamlit run house_price_app.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .block-container { padding: 2rem 3rem; }

    .hero-box {
        background: linear-gradient(135deg, #1a3a5c 0%, #1e6091 100%);
        border-radius: 14px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 2rem;
    }
    .hero-box h1 { font-size: 2rem; font-weight: 700; margin: 0; }
    .hero-box p  { font-size: 0.95rem; opacity: 0.85; margin: 0.4rem 0 0 0; }

    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e8ecf0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .metric-label { font-size: 0.78rem; color: #6b7280; font-weight: 500; margin-bottom: 4px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1e6091; }
    .metric-model { font-size: 0.72rem; color: #9ca3af; margin-top: 2px; }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1a3a5c;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #e8ecf0;
    }

    .stDataFrame { border-radius: 10px; }

    div[data-testid="stSelectbox"] > label,
    div[data-testid="stFileUploader"] > label {
        font-weight: 600;
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)


# ── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <div class="badge">INTERNSHIP PROJECT · T-8 · SYNENT TECHNOLOGY</div>
    <h1>🏠 House Price Prediction</h1>
    <p>Predicting residential property prices using Linear Regression & Random Forest on the King County, WA dataset.</p>
</div>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    st.markdown("### Model Settings")
    test_size      = st.slider("Test Split Size", 0.1, 0.4, 0.2, 0.05)
    n_estimators   = st.slider("RF — Number of Trees", 50, 300, 100, 50)
    random_state   = st.number_input("Random State", value=42, step=1)

    st.markdown("---")
    run_btn = st.button("▶ Run Analysis", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("<small>Tech Stack: pandas · sklearn · seaborn · streamlit</small>", unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file:
    df = load_data(uploaded_file)
else:
    try:
        df = pd.read_csv(r"E:\Internship\Synent Technology\T-8\kc_house_data.csv")
    except FileNotFoundError:
        st.warning("⚠️ Default CSV not found. Please upload your dataset using the sidebar.")
        st.stop()


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data Overview",
    "📊 Visualizations",
    "🤖 Model Results",
    "🔮 Predict Price"
])


# ── TAB 1 : DATA OVERVIEW ─────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",    f"{df.shape[0]:,}")
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())
    c4.metric("Price Range",   f"${df['price'].min()/1e6:.2f}M – ${df['price'].max()/1e6:.2f}M")

    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Data Types</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame(df.dtypes, columns=["Type"]).reset_index()
        dtype_df.columns = ["Column", "Type"]
        st.dataframe(dtype_df, use_container_width=True, height=300)

    with col2:
        st.markdown('<div class="section-title">Null Values</div>', unsafe_allow_html=True)
        null_df = pd.DataFrame(df.isnull().sum(), columns=["Null Count"]).reset_index()
        null_df.columns = ["Column", "Null Count"]
        st.dataframe(null_df, use_container_width=True, height=300)

    st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)


# ── TAB 2 : VISUALIZATIONS ────────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Price Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df['price'], bins=40, color='#1e6091', edgecolor='white', ax=ax)
        ax.set_xlabel("Price (USD)")
        ax.set_ylabel("Frequency")
        ax.set_title("House Price Distribution", fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown('<div class="section-title">Price vs sqft_living</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.scatter(df['sqft_living'], df['price'], alpha=0.3, s=8, color='#1e6091')
        ax2.set_xlabel("Living Area (sqft)")
        ax2.set_ylabel("Price (USD)")
        ax2.set_title("Price vs Living Area", fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    num_df = df.select_dtypes(include=[np.number])
    sns.heatmap(
        num_df.corr(),
        annot=True, fmt='.2f',
        cmap='coolwarm',
        linewidths=0.5,
        annot_kws={'size': 7},
        ax=ax3
    )
    ax3.set_title("Correlation Heatmap", fontweight='bold', fontsize=13)
    fig3.tight_layout()
    st.pyplot(fig3)


# ── TRAIN MODELS (shared across tabs) ─────────────────────────────────────────
@st.cache_resource
def train_models(test_sz, n_est, r_state):
    data = df.copy()
    data.drop(['id', 'date'], axis=1, inplace=True, errors='ignore')

    X = data.drop('price', axis=1)
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_sz, random_state=r_state
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    rf = RandomForestRegressor(n_estimators=n_est, random_state=r_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    return lr, rf, X, X_train, X_test, y_train, y_test, y_pred_lr, y_pred_rf

lr_model, rf_model, X, X_train, X_test, y_train, y_test, y_pred_lr, y_pred_rf = train_models(
    test_size, n_estimators, int(random_state)
)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)


# ── TAB 3 : MODEL RESULTS ─────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LR — R² Score",  f"{r2_lr:.4f}")
    c2.metric("LR — RMSE",      f"${rmse_lr:,.0f}")
    c3.metric("RF — R² Score",  f"{r2_rf:.4f}", delta=f"+{r2_rf - r2_lr:.4f} vs LR")
    c4.metric("RF — RMSE",      f"${rmse_rf:,.0f}", delta=f"-${rmse_lr - rmse_rf:,.0f} vs LR", delta_color="inverse")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Feature Importances — Top 10</div>', unsafe_allow_html=True)
        fi = pd.DataFrame({
            'Feature':    X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)

        fig4, ax4 = plt.subplots(figsize=(7, 5))
        sns.barplot(x='Importance', y='Feature', data=fi, palette='Blues_r', ax=ax4)
        ax4.set_title("Top 10 Feature Importances", fontweight='bold')
        ax4.set_xlabel("Importance Score")
        fig4.tight_layout()
        st.pyplot(fig4)

    with col2:
        st.markdown('<div class="section-title">Actual vs Predicted — Random Forest</div>', unsafe_allow_html=True)
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        ax5.scatter(y_test, y_pred_rf, alpha=0.35, s=10, color='#1e6091')
        ax5.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--', linewidth=2, label='Perfect Prediction'
        )
        ax5.set_xlabel("Actual Price (USD)")
        ax5.set_ylabel("Predicted Price (USD)")
        ax5.set_title("Actual vs Predicted", fontweight='bold')
        ax5.legend()
        ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
        fig5.tight_layout()
        st.pyplot(fig5)

    st.markdown('<div class="section-title">Model Comparison Table</div>', unsafe_allow_html=True)
    comparison = pd.DataFrame({
        'Model' : ['Linear Regression', 'Random Forest'],
        'R² Score': [round(r2_lr, 4), round(r2_rf, 4)],
        'RMSE ($)': [f"${rmse_lr:,.0f}", f"${rmse_rf:,.0f}"],
        'Winner' : ['', '✅']
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)


# ── TAB 4 : PREDICT PRICE ─────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">🔮 Predict a House Price</div>', unsafe_allow_html=True)
    st.write("Enter house features below and get a price estimate from the Random Forest model.")

    col1, col2, col3 = st.columns(3)

    with col1:
        bedrooms   = st.number_input("Bedrooms",         1,  10,  3)
        bathrooms  = st.number_input("Bathrooms",        1,  8,   2)
        sqft_living= st.number_input("Sqft Living",      500,15000,1800)
        sqft_lot   = st.number_input("Sqft Lot",         500,100000,5000)
        floors     = st.number_input("Floors",           1,  4,   1)

    with col2:
        waterfront = st.selectbox("Waterfront",   [0, 1])
        view       = st.slider("View (0-4)",       0, 4, 0)
        condition  = st.slider("Condition (1-5)",  1, 5, 3)
        grade      = st.slider("Grade (1-13)",     1, 13, 7)
        sqft_above = st.number_input("Sqft Above", 500, 10000, 1500)

    with col3:
        sqft_basement = st.number_input("Sqft Basement", 0, 5000, 0)
        yr_built      = st.number_input("Year Built",    1900, 2015, 1990)
        yr_renovated  = st.number_input("Year Renovated",0, 2015, 0)
        zipcode       = st.number_input("Zipcode",       98001, 98199, 98052)
        lat           = st.number_input("Latitude",      47.0, 48.0, 47.5, format="%.4f")
        long          = st.number_input("Longitude",    -122.5,-121.0,-122.0, format="%.4f")
        sqft_living15 = st.number_input("Sqft Living15", 500, 10000, 1800)
        sqft_lot15    = st.number_input("Sqft Lot15",    500, 100000, 5000)

    if st.button("🏷️ Predict Price", type="primary"):
        input_data = pd.DataFrame([{
            'bedrooms': bedrooms, 'bathrooms': bathrooms,
            'sqft_living': sqft_living, 'sqft_lot': sqft_lot,
            'floors': floors, 'waterfront': waterfront,
            'view': view, 'condition': condition,
            'grade': grade, 'sqft_above': sqft_above,
            'sqft_basement': sqft_basement, 'yr_built': yr_built,
            'yr_renovated': yr_renovated, 'zipcode': zipcode,
            'lat': lat, 'long': long,
            'sqft_living15': sqft_living15, 'sqft_lot15': sqft_lot15
        }])

        predicted_price = rf_model.predict(input_data)[0]

        st.success(f"### 🏠 Estimated Price: **${predicted_price:,.0f}**")
        st.caption("Prediction powered by Random Forest (100 trees) trained on King County, WA data.")