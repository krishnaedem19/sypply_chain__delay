# app.py — Supply Chain Delay Prediction (Streamlit 1.40+ compatible)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix

# ============================================================
# 🎨 PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Supply Chain Delay Prediction",
    page_icon="📦",
    layout="wide"
)

# ============================================================
# 🧭 SIDEBAR
# ============================================================
st.sidebar.title("📦 Supply Chain Delay Prediction")
st.sidebar.markdown("---")

# ============================================================
# 🧹 DATA LOAD & PREPROCESSING
# ============================================================
@st.cache_data
def load_raw():
    path = "data/supply_chain_data.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="unicode_escape", low_memory=False)
        return df
    return None


def basic_preprocess(df: pd.DataFrame):
    """Minimal cleanup and encoding."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Late_delivery_risk" not in df.columns:
        if "Days for shipping (real)" in df.columns and "Days for shipment (scheduled)" in df.columns:
            df["Late_delivery_risk"] = np.where(
                df["Days for shipping (real)"] > df["Days for shipment (scheduled)"], 1, 0
            )
        else:
            df["Late_delivery_risk"] = np.nan

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).replace(["?", "nan", "None"], np.nan)

    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    return df


def get_features_targets(df: pd.DataFrame):
    candidates = [
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Order Item Quantity",
        "Order Item Discount",
        "Product Price",
        "Order Region",
        "Order State",
        "Order City",
        "Category Name",
        "Department Name",
        "Shipping Mode",
    ]
    features = [c for c in candidates if c in df.columns]
    target = "Late_delivery_risk" if "Late_delivery_risk" in df.columns else None
    return features, target


# ============================================================
# ⚙️ MODEL UTILITIES
# ============================================================
MODEL_PATH = "delay_model.pkl"
METRICS_PATH = "model_metrics.pkl"


class LabelEncoderPipelineFriendly(LabelEncoder):
    """LabelEncoder wrapper that handles unknown labels safely."""

    def fit(self, X, y=None):
        super().fit(X)
        return self

    def transform(self, X):
        X = pd.Series(X)
        return X.map(lambda s: self.classes_.tolist().index(s) if s in self.classes_ else -1).values

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def train_model(df, features, target):
    df = df.copy()
    df = df[features + [target]].dropna()
    X = df[features]
    y = df[target]

    cat_features = [f for f in features if X[f].dtype == "object"]
    num_features = [f for f in features if X[f].dtype != "object"]

    cat_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")),
               ("encoder", LabelEncoderPipelineFriendly())]
    )
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=500))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)
    joblib.dump({"accuracy": acc, "cm": cm.tolist()}, METRICS_PATH)

    return acc, cm


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def load_metrics():
    if os.path.exists(METRICS_PATH):
        return joblib.load(METRICS_PATH)
    return None


# ============================================================
# 🚀 DASHBOARD
# ============================================================
st.title("📦 Supply Chain Delay Prediction Dashboard")
st.markdown("Predict shipment delays, analyze performance, and optimize logistics.")
st.markdown("---")

df = load_raw()
if df is None:
    st.error("❌ File `supply_chain_data.csv` not found.")
    st.stop()

df = basic_preprocess(df)
features, target = get_features_targets(df)

# ============================================================
# 🔍 FILTERS
# ============================================================
with st.expander("🔎 Data Filters"):
    col1, col2, col3 = st.columns(3)
    region_filter = col1.multiselect("Region", options=sorted(df["Order Region"].dropna().unique()) if "Order Region" in df.columns else [])
    dept_filter = col2.multiselect("Department", options=sorted(df["Department Name"].dropna().unique()) if "Department Name" in df.columns else [])
    ship_filter = col3.multiselect("Shipping Mode", options=sorted(df["Shipping Mode"].dropna().unique()) if "Shipping Mode" in df.columns else [])

    filtered = df.copy()
    if region_filter:
        filtered = filtered[filtered["Order Region"].isin(region_filter)]
    if dept_filter:
        filtered = filtered[filtered["Department Name"].isin(dept_filter)]
    if ship_filter:
        filtered = filtered[filtered["Shipping Mode"].isin(ship_filter)]

# ============================================================
# 📊 KPI SUMMARY
# ============================================================
col1, col2, col3, col4 = st.columns(4)
total_orders = len(filtered)
avg_discount = filtered["Order Item Discount"].mean() if "Order Item Discount" in filtered.columns else np.nan
avg_price = filtered["Product Price"].mean() if "Product Price" in filtered.columns else np.nan
late_rate = filtered["Late_delivery_risk"].mean() if "Late_delivery_risk" in filtered.columns else np.nan

col1.metric("Total Orders", f"{total_orders:,}")
col2.metric("Avg Discount", f"{avg_discount:.2f}")
col3.metric("Avg Price", f"{avg_price:.2f}")
col4.metric("Late Delivery Rate", f"{late_rate:.1%}")

# ============================================================
# 📈 CHARTS
# ============================================================
if "Order Region" in filtered.columns and "Late_delivery_risk" in filtered.columns:
    fig1 = px.histogram(filtered, x="Order Region", color="Late_delivery_risk",
                        barmode="group", title="Late Deliveries by Region")
    st.plotly_chart(fig1, width="stretch")

if "Shipping Mode" in filtered.columns and "Late_delivery_risk" in filtered.columns:
    fig2 = px.histogram(filtered, x="Shipping Mode", color="Late_delivery_risk",
                        title="Late Deliveries by Shipping Mode")
    st.plotly_chart(fig2, width="stretch")

if "Order Date" in filtered.columns and "Late_delivery_risk" in filtered.columns:
    temp = filtered.copy()
    temp["Order Date"] = pd.to_datetime(temp["Order Date"], errors="coerce")
    temp = temp.dropna(subset=["Order Date"])
    trend = temp.groupby(pd.Grouper(key="Order Date", freq="M"))["Late_delivery_risk"].mean().reset_index()
    fig3 = px.line(trend, x="Order Date", y="Late_delivery_risk",
                   title="Monthly Late Delivery Trend", markers=True)
    st.plotly_chart(fig3, width="stretch")

# ============================================================
# 🧠 MODEL TRAINING / PREDICTION
# ============================================================
st.markdown("### 🧠 Train or Use Predictive Model")

model_loaded = load_model()
metrics = load_metrics()

if st.button("🔁 Train Model"):
    with st.spinner("Training model..."):
        if target and len(features) >= 2:
            acc, cm = train_model(df, features, target)
            st.success(f"✅ Model trained! Accuracy: {acc:.2%}")
        else:
            st.warning("Not enough features or target missing for training.")

if model_loaded is not None:
    st.success("✅ Model loaded and ready.")
    if metrics:
        st.write(f"**Last Model Accuracy:** {metrics['accuracy']:.2%}")
        st.write("**Confusion Matrix:**", metrics["cm"])

    st.markdown("---")
    st.subheader("📦 Predict a New Order")

    input_data = {}
    for feat in features:
        if df[feat].dtype == "object":
            input_data[feat] = st.selectbox(feat, options=sorted(df[feat].dropna().unique()))
        else:
            input_data[feat] = st.number_input(feat, value=float(df[feat].mean()) if df[feat].notna().any() else 0.0)

    if st.button("🚀 Predict Delay"):
        X_new = pd.DataFrame([input_data])
        pred = model_loaded.predict(X_new)[0]
        if pred == 1:
            st.error("⚠️ Prediction: **Likely Late Delivery**")
        else:
            st.success("✅ Prediction: **On Time Delivery**")
else:
    st.info("Train or upload a model to enable predictions.")

# ============================================================
# 📋 RAW DATA
# ============================================================
with st.expander("📋 View Raw Data"):
    st.dataframe(filtered.head(500), width="stretch")

# ============================================================
# 🔧 OPTIMIZATION RECOMMENDATIONS
# ============================================================
st.markdown("---")
st.subheader("🔧 Supply Chain Optimization Insights")

if "Shipping Mode" in filtered.columns and "Late_delivery_risk" in filtered.columns:
    risk_by_mode = filtered.groupby("Shipping Mode")["Late_delivery_risk"].mean().sort_values()
    st.write("### Late Risk by Shipping Mode")
    st.bar_chart(risk_by_mode)

    best_mode = risk_by_mode.idxmin()
    st.success(f"✅ **Recommendation:** Prefer `{best_mode}` shipping for lower delay risk.")
