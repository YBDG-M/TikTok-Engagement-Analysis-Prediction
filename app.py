# ==========================================
# TikTok / Instagram Engagement Analysis - Streamlit
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import joblib

# -----------------------------
# 1. Setup Streamlit
# -----------------------------
st.set_page_config(
    page_title="Engagement Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Engagement Analysis & Prediction")
st.markdown("Aplikasi ini menganalisis data postingan untuk memahami **engagement per follower**.")

# -----------------------------
# 2. Upload Data
# -----------------------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV (TikTok/Instagram)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # -----------------------------
    # 3. Preprocessing
    # -----------------------------
    columns_needed = [
        'authorMeta/fans', 'authorMeta/name', 'collectCount',
        'commentCount', 'createTimeISO', 'diggCount', 'isSponsored',
        'locationMeta/city', 'locationMeta/locationName', 'shareCount',
        'text', 'videoMeta/duration', 'webVideoUrl', 'playCount'
    ]
    df = df[columns_needed].copy()

    # Hapus missing values teks
    df.dropna(subset=["text"], inplace=True)

    # Isi missing value numerik dengan 0
    numeric_cols = [
        'collectCount', 'commentCount', 'diggCount',
        'shareCount', 'playCount', 'authorMeta/fans', 'videoMeta/duration'
    ]
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Isi string kosong
    string_cols = [
        'authorMeta/name', 'locationMeta/city',
        'locationMeta/locationName', 'webVideoUrl'
    ]
    df[string_cols] = df[string_cols].fillna("Unknown")

    # Boolean
    df['isSponsored'] = df['isSponsored'].fillna(False)

    # -----------------------------
    # 4. Feature Engineering
    # -----------------------------
    df['timestamp'] = pd.to_datetime(df['createTimeISO'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.day_name()

    # Engagement metrics
    df['likes_per_follower']    = df['diggCount'] / (df['authorMeta/fans'] + 1)
    df['comments_per_follower'] = df['commentCount'] / (df['authorMeta/fans'] + 1)
    df['comments_per_like']     = df['commentCount'] / (df['diggCount'] + 1)
    df['shares_per_follower']   = df['shareCount'] / (df['authorMeta/fans'] + 1)
    df['saves_per_follower']    = df['collectCount'] / (df['authorMeta/fans'] + 1)
    df['plays_per_follower']    = df['playCount'] / (df['authorMeta/fans'] + 1)

    df['total_engagement_per_follower'] = (
        df['diggCount'] + df['commentCount'] + df['shareCount'] + df['collectCount']
    ) / (df['authorMeta/fans'] + 1)

    df['engagement_rate'] = (
        df['diggCount'] + df['commentCount'] + df['shareCount'] + df['collectCount']
    ) / (df['playCount'] + 1) * 100

    st.subheader("👀 Preview Data")
    st.dataframe(df.head())

    # -----------------------------
    # 5. EDA
    # -----------------------------
    st.header("🔎 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs([
        "Per Jam", "Per Hari", "Heatmap Hari vs Jam"
    ])

    with tab1:
        hourly_engagement = df.groupby('hour')['total_engagement_per_follower'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x='hour', y='total_engagement_per_follower', data=hourly_engagement, palette='viridis', ax=ax)
        ax.set_title("Rata-rata Engagement per Follower per Jam", fontsize=14, weight="bold")
        st.pyplot(fig)

    with tab2:
        days_order_en = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        days_order_id = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']
        daily_engagement = (
            df.groupby('dayofweek')['total_engagement_per_follower']
            .mean()
            .reindex(days_order_en)
            .reset_index()
        )
        daily_engagement['dayofweek'] = days_order_id

        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x='dayofweek', y='total_engagement_per_follower', data=daily_engagement, palette='magma', ax=ax)
        ax.set_title("Rata-rata Engagement per Follower per Hari", fontsize=14, weight="bold")
        st.pyplot(fig)

    with tab3:
        heatmap_data = df.pivot_table(
            index="hour",
            columns="dayofweek",
            values="total_engagement_per_follower",
            aggfunc="mean"
        )
        heatmap_data = heatmap_data[days_order_en]
        heatmap_data.columns = days_order_id

        fig, ax = plt.subplots(figsize=(12,6))
        sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax, cbar_kws={'label': 'Engagement per Follower'})
        ax.set_title("Heatmap Engagement per Follower (Hari vs Jam)", fontsize=14, weight="bold")
        st.pyplot(fig)

    # -----------------------------
    # 6. Model Training
    # -----------------------------
    st.header("🤖 Model Training & Evaluation")

    features = [
        "likes_per_follower", "comments_per_follower",
        "shares_per_follower", "saves_per_follower",
        "plays_per_follower", "hour"
    ]
    X = df[features]
    y = df["total_engagement_per_follower"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    st.metric("MAE", f"{mae:.4f}")
    st.metric("R² Score", f"{r2:.4f}")
    st.write("Cross-Validation R² Scores:", cv_scores)
    st.write("Mean CV R²:", round(cv_scores.mean(), 4))

    # -----------------------------
    # 7. Feature Importance
    # -----------------------------
    st.subheader("📌 Feature Importance")
    importances = model.feature_importances_
    importances_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x="Importance", y="Feature", data=importances_df, palette="viridis", ax=ax)
    ax.set_title("Feature Importance RandomForest", fontsize=14, weight="bold")
    st.pyplot(fig)
    st.dataframe(importances_df)

    # -----------------------------
    # 8. SHAP Analysis
    # -----------------------------
    st.subheader("📊 SHAP Analysis")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    st.markdown("**SHAP Summary Plot (bar)**")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)

    st.markdown("**SHAP Beeswarm Plot**")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)

    # -----------------------------
    # 9. Save Model
    # -----------------------------
    joblib.dump(model, "engagement_model.pkl")
    st.success("✅ Model berhasil disimpan sebagai `engagement_model.pkl`")

else:
    st.warning("👆 Silakan upload file dataset (CSV) terlebih dahulu.")
