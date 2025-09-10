# app.py (enhanced)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="🏡 Housing Price Predictor", layout="wide", initial_sidebar_state="expanded")

# --- Paths (adjust if needed) ---
ROOT = os.path.dirname(os.path.abspath(__file__))  # project root dynamically
MODEL_TUNED = os.path.join(ROOT, "models", "final_model_tuned.joblib")
MODEL_SIMPLE = os.path.join(ROOT, "models", "final_model.joblib")
DATA_CSV = os.path.join(ROOT, "data", "train.csv")
LOGO_PATH = os.path.join(ROOT, "assets", "logo.png")  # optional logo (create assets/logo.png if you want)

# --- Styling (small CSS tweaks) ---
st.markdown(
    """
    <style>
    /* page background soft */
    .stApp {
        background-color: #0b1020;
        color: #eef2f7;
    }
    /* Title center */
    .title {
        text-align: left;
        padding-bottom: 6px;
    }
    /* Card like preview */
    .stDataFrame table {
        border-radius: 8px;
        overflow: hidden;
    }
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
    }
    /* Small visual polish for containers */
    .css-1d391kg { padding: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Utils: load model and training defaults (cached) ---
@st.cache_data
def load_model():
    path = MODEL_TUNED if os.path.exists(MODEL_TUNED) else MODEL_SIMPLE
    if not os.path.exists(path):
        return None, f"No model file found at {MODEL_TUNED} or {MODEL_SIMPLE}"
    # joblib.load may raise if sklearn versions mismatch — let the exception propagate so we can show an error
    model = joblib.load(path)
    return model, None

@st.cache_data
def load_train_defaults():
    if not os.path.exists(DATA_CSV):
        return None
    df = pd.read_csv(DATA_CSV)
    df = df.copy()
    if 'SalePrice' in df.columns:
        df = df.drop(columns=['SalePrice'])
    num_medians = df.select_dtypes(include=[np.number]).median().to_dict()
    cat_modes = df.select_dtypes(include=['object','category']).mode().iloc[0].to_dict() if not df.select_dtypes(include=['object','category']).empty else {}
    return {'num_medians': num_medians, 'cat_modes': cat_modes, 'columns': df.columns.tolist(), 'sample_head': df.head(3)}

# --- Load model & defaults ---
model, err = load_model()
defaults = load_train_defaults()

# --- Header: optional logo + title + subtitle ---
header_col1, header_col2 = st.columns([0.12, 0.88])
with header_col1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=90)
with header_col2:
    st.markdown("<h1 class='title'>🏡 Housing Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("**Upload your CSV and sip your tea ☕ while the app generates insights & predictions.**")

st.markdown("---")

# --- If model missing, show helpful message and stop ---
if err:
    st.error(err)
    st.info("Run your training script to produce models/final_model.joblib or models/final_model_tuned.joblib, then refresh.")
    st.stop()

# --- Sidebar instructions & mode selection ---
st.sidebar.header("How to use")
st.sidebar.markdown(
    """
1. Upload a CSV with the same features used for training (or use the sample).
2. View EDA, charts and predictions.
3. Download predictions as CSV.
"""
)
mode = st.sidebar.radio("Input mode", ["Upload CSV (recommended)", "Single input (quick)"])

# --- Helper plotting functions (small and robust) ---
def plot_hist(series, title="Distribution"):
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=90)
    sns.histplot(series.dropna(), kde=True, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def plot_heatmap(df_subset, title="Correlation heatmap"):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=90)
    sns.heatmap(df_subset.corr(), annot=True, fmt=".2f", cmap="RdYlBu", ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def plot_box(df, xcol, ycol, title=None):
    fig, ax = plt.subplots(figsize=(10, 4), dpi=90)
    sns.boxplot(data=df, x=xcol, y=ycol, ax=ax)
    ax.set_title(title or f"{ycol} by {xcol}")
    plt.xticks(rotation=90)
    st.pyplot(fig)

def scatter_plot(df, xcol, ycol, title=None):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=90)
    sns.scatterplot(data=df, x=xcol, y=ycol, ax=ax)
    ax.set_title(title or f"{ycol} vs {xcol}")
    st.pyplot(fig)

# --- Upload CSV mode ---
if mode == "Upload CSV (recommended)":
    st.markdown("### Upload CSV and explore")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        # Read CSV
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        # Top-level preview and basic info in tabs
        tab_preview, tab_summary, tab_visuals, tab_predictions = st.tabs(
            ["📋 Preview", "🧾 Summary", "📈 Visuals", "🤖 Predictions"]
        )

        # --- Preview tab ---
        with tab_preview:
            st.write("Preview of uploaded data")
            st.dataframe(df_in.head(10), use_container_width=True)
            st.write(f"Shape: **{df_in.shape[0]} rows × {df_in.shape[1]} columns**")
            if st.checkbox("Show full columns list"):
                st.write(df_in.columns.tolist())

        # --- Summary tab ---
        with tab_summary:
            st.subheader("Dataset Summary")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write("**Missing values (top 20)**")
                missing = df_in.isnull().sum().sort_values(ascending=False)
                missing = missing[missing > 0]
                if not missing.empty:
                    st.dataframe(missing.head(20))
                else:
                    st.write("No missing values detected.")
            with col2:
                st.write("**Numeric feature summary (.describe().T)**")
                try:
                    desc = df_in.describe().T
                    st.dataframe(desc)
                except Exception as e:
                    st.write("Could not compute describe():", e)

            # quick top-5 categorical distributions (if exist)
            cat_cols = df_in.select_dtypes(include=['object','category']).columns.tolist()
            if cat_cols:
                st.write("**Sample categorical value counts (top 3 columns)**")
                for c in cat_cols[:3]:
                    vc = df_in[c].value_counts().head(10)
                    st.write(f"**{c}**")
                    st.dataframe(vc)

        # --- Visuals tab ---
        with tab_visuals:
            st.subheader("Automatic Visualizations")
            # If SalePrice present: distribution, heatmap, boxplots, scatter
            if "SalePrice" in df_in.columns:
                with st.expander("Target: SalePrice"):
                    try:
                        plot_hist(df_in["SalePrice"], title="SalePrice distribution")
                        # log transform chart
                        plot_hist(np.log1p(df_in["SalePrice"]), title="Log(1 + SalePrice) distribution")
                    except Exception as e:
                        st.write("Could not plot SalePrice distribution:", e)

                # Correlation heatmap for top correlated numeric features
                try:
                    numeric_corr = df_in.corr(numeric_only=True)
                    if "SalePrice" in numeric_corr.columns:
                        top_feats = numeric_corr["SalePrice"].abs().sort_values(ascending=False).head(11).index.tolist()
                        df_sub = df_in[top_feats].select_dtypes(include=[np.number])
                        st.write("Correlation heatmap (top features with SalePrice)")
                        plot_heatmap(df_sub, title="Correlation (top features)")
                except Exception as e:
                    st.write("Could not compute correlation heatmap:", e)

                # Boxplot by Neighborhood if present and not too many categories
                try:
                    if "Neighborhood" in df_in.columns and df_in["Neighborhood"].nunique() < 30:
                        st.write("SalePrice by Neighborhood (boxplot)")
                        plot_box(df_in, xcol="Neighborhood", ycol="SalePrice", title="SalePrice by Neighborhood")
                except Exception as e:
                    st.write("Could not plot Neighborhood boxplot:", e)

                # Scatterplots with top 2-3 correlated features
                try:
                    if "SalePrice" in numeric_corr.columns:
                        top_x = numeric_corr["SalePrice"].abs().sort_values(ascending=False).index.tolist()[1:4]
                        for xcol in top_x:
                            if xcol in df_in.columns:
                                scatter_plot(df_in, xcol=xcol, ycol="SalePrice", title=f"SalePrice vs {xcol}")
                except Exception as e:
                    st.write("Could not create scatterplots:", e)
            else:
                st.info("No 'SalePrice' column found — visuals limited to general distributions.")
                # General numeric distribution example
                numcols = df_in.select_dtypes(include=[np.number]).columns.tolist()
                if numcols:
                    sample_col = numcols[0]
                    try:
                        plot_hist(df_in[sample_col], title=f"Distribution of {sample_col}")
                    except Exception as e:
                        st.write("Could not plot numeric distribution:", e)

        # --- Predictions tab ---
        with tab_predictions:
            st.subheader("Predictions")
            # Show spinner while computing predictions
            with st.spinner("Crunching numbers... brewing predictions ☕"):
                try:
                    preds_log = model.predict(df_in)
                    preds_price = np.expm1(preds_log)
                    out = df_in.copy()
                    out["predicted_price"] = preds_price
                    st.success("Predictions ready 🎉")
                    st.dataframe(out.head(10), use_container_width=True)
                    csv = out.to_csv(index=False).encode('utf-8')
                    st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        # After tabs: show approx feature importances below (if available)
        st.markdown("---")
        st.header("Model info & feature importances (approx)")
        try:
            if hasattr(model, 'named_steps'):
                m = model.named_steps.get('model', None)
                pre = model.named_steps.get('preproc', None)
            else:
                m = model
                pre = None

            if m is not None and hasattr(m, 'feature_importances_'):
                fi = m.feature_importances_
                feat_names = None
                if pre is not None:
                    try:
                        # Try to extract feature names from ColumnTransformer/OHE
                        num_cols = pre.transformers_[0][2]
                        cat_ohe = pre.transformers_[1][1].named_steps['ohe']
                        cat_cols = pre.transformers_[1][2]
                        try:
                            cat_names = cat_ohe.get_feature_names_out(cat_cols).tolist()
                        except Exception:
                            cat_names = []
                        feat_names = list(num_cols) + list(cat_names)
                    except Exception:
                        feat_names = [f"f{i}" for i in range(len(fi))]
                imp_df = pd.DataFrame({'feature': feat_names, 'importance': fi}).sort_values('importance', ascending=False).head(30)
                st.dataframe(imp_df)
            else:
                st.info("Feature importances not available for this model type.")
        except Exception as ex:
            st.write("Could not extract feature importances:", ex)

# --- Single-input quick mode ---
else:
    st.markdown("### Single input (quick) — fill a few fields and predict")
    if defaults is None:
        st.error("Training CSV not found; single input requires the training CSV to infer defaults. Upload a CSV instead.")
    else:
        cols = defaults['columns']
        # Expose a few useful fields
        overall_qual = st.slider("Overall Quality (OverallQual, 1-10)", 1, 10, int(defaults['num_medians'].get('OverallQual', 6)))
        gr_liv_area = st.number_input("GrLivArea (Living area sqft)", min_value=100, max_value=10000, value=int(defaults['num_medians'].get('GrLivArea', 1500)))
        year_built = st.number_input("YearBuilt", min_value=1800, max_value=2025, value=int(defaults['num_medians'].get('YearBuilt', 1970)))
        neighborhood = None
        if 'Neighborhood' in defaults['cat_modes']:
            neighborhood = st.selectbox("Neighborhood (approx)", options=[defaults['cat_modes'].get('Neighborhood')])

        # build a single-row DataFrame with defaults then overwrite the exposed fields
        row = {}
        for c in cols:
            if c in defaults['num_medians']:
                row[c] = defaults['num_medians'][c]
            elif c in defaults['cat_modes']:
                row[c] = defaults['cat_modes'][c]
            else:
                row[c] = 0 if c in defaults['num_medians'] else "missing"

        if 'OverallQual' in cols: row['OverallQual'] = overall_qual
        if 'GrLivArea' in cols: row['GrLivArea'] = gr_liv_area
        if 'YearBuilt' in cols: row['YearBuilt'] = year_built
        if 'Neighborhood' in cols and neighborhood is not None: row['Neighborhood'] = neighborhood

        input_df = pd.DataFrame([row], columns=cols)
        st.write("Input features (preview)")
        st.dataframe(input_df.head())

        if st.button("Predict single record"):
            with st.spinner("Predicting..."):
                try:
                    pred_log = model.predict(input_df)[0]
                    pred_price = np.expm1(pred_log)
                    st.success(f"Predicted Sale Price: ${pred_price:,.0f}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # Show model info / importances here as well
    st.markdown("---")
    st.header("Model info & feature importances (approx)")
    try:
        if hasattr(model, 'named_steps'):
            m = model.named_steps.get('model', None)
            pre = model.named_steps.get('preproc', None)
        else:
            m = model
            pre = None

        if m is not None and hasattr(m, 'feature_importances_'):
            fi = m.feature_importances_
            feat_names = None
            if pre is not None:
                try:
                    num_cols = pre.transformers_[0][2]
                    cat_ohe = pre.transformers_[1][1].named_steps['ohe']
                    cat_cols = pre.transformers_[1][2]
                    try:
                        cat_names = cat_ohe.get_feature_names_out(cat_cols).tolist()
                    except Exception:
                        cat_names = []
                    feat_names = list(num_cols) + list(cat_names)
                except Exception:
                    feat_names = [f"f{i}" for i in range(len(fi))]
            imp_df = pd.DataFrame({'feature': feat_names, 'importance': fi}).sort_values('importance', ascending=False).head(30)
            st.dataframe(imp_df)
        else:
            st.info("Feature importances not available for this model type.")
    except Exception as ex:
        st.write("Could not extract feature importances:", ex)
