# etf_recommender_checkbox.py
# Streamlit-based ETF Recommendation System with PCA visualization (only recommended ETFs labeled)

import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

st.set_page_config(page_title="ETF Recommendation System", layout="wide")

st.title("📈 ETF Recommendation System (Checkbox Selection)")

# --- Helper: Clean and load CSV ---
def load_etf_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.replace('-', '', regex=False)
                .replace('', np.nan)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['close']).reset_index(drop=True)

# --- Helper: Compute features for each ETF ---
def compute_features(df):
    s = {}
    df['ret'] = df['close'].pct_change()
    daily = df['ret'].dropna()
    if daily.empty:
        return None

    TRADING_DAYS = 252
    mean_daily = daily.mean()
    vol_daily = daily.std()
    s['ann_return'] = (1 + mean_daily) ** TRADING_DAYS - 1
    s['ann_vol'] = vol_daily * np.sqrt(TRADING_DAYS)
    s['sharpe'] = mean_daily / (vol_daily + 1e-9)
    s['skew'] = skew(daily)
    s['kurtosis'] = kurtosis(daily)

    # max drawdown
    cum = (1 + daily).cumprod()
    peak = cum.cummax()
    drawdown = (cum / peak - 1)
    s['max_drawdown'] = drawdown.min()

    # momentum (3, 6, 12 months)
    for m, days in [('mom_3m', 63), ('mom_6m', 126), ('mom_12m', 252)]:
        if len(df) > days:
            s[m] = df['close'].iloc[-1] / df['close'].iloc[-days] - 1
        else:
            s[m] = np.nan

    # liquidity
    if 'volume' in df.columns:
        s['avg_volume'] = df['volume'].dropna().tail(252).mean()
    else:
        s['avg_volume'] = np.nan

    return s

# --- Load ETFs ---
data_dir = st.text_input("Enter folder path containing your ETF CSVs:", "")
etf_data = {}
if data_dir and os.path.isdir(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            etf_data[name] = load_etf_csv(f)
        except Exception as e:
            st.warning(f"Skipping {name}: {e}")
    st.success(f"✅ Loaded {len(etf_data)} ETFs")

if etf_data:
    # Compute features for each ETF
    feats = {name: compute_features(df) for name, df in etf_data.items()}
    feats = {k: v for k, v in feats.items() if v is not None}
    features_df = pd.DataFrame(feats).T.fillna(features_df.median() if 'features_df' in locals() else 0)

    # Standardize + PCA
    X = features_df.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)

    # KNN model
    knn = NearestNeighbors(n_neighbors=6, metric='euclidean').fit(Xp)

    # ETF selection (checkbox list)
    st.subheader("Select one or more ETFs to get recommendations:")
    selected = st.multiselect("Choose ETF(s):", list(features_df.index))

    if selected:
        recs_all = []
        for etf_name in selected:
            if etf_name not in features_df.index:
                continue
            idx = list(features_df.index).index(etf_name)
            vec = Xp[idx].reshape(1, -1)
            dists, idxs = knn.kneighbors(vec, n_neighbors=6)
            for d, i in zip(dists[0], idxs[0]):
                name = features_df.index[i]
                if name != etf_name:
                    recs_all.append({'ETF': name, 'Distance': float(d)})
        recs_df = pd.DataFrame(recs_all).drop_duplicates().sort_values("Distance").head(10)
        st.dataframe(recs_df, use_container_width=True)

        # --- PCA visualization ---
        st.subheader("PCA Visualization (Recommended ETFs labeled)")
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot all ETFs (gray)
        ax.scatter(Xp[:, 0], Xp[:, 1], color='lightgray', alpha=0.5, s=30)

        # Highlight selected ETFs (blue)
        sel_idx = [list(features_df.index).index(s) for s in selected]
        ax.scatter(Xp[sel_idx, 0], Xp[sel_idx, 1], color='blue', s=80, label='Selected')

        # Highlight recommended ETFs (red + label)
        rec_idx = [list(features_df.index).index(r) for r in recs_df['ETF'] if r in features_df.index]
        ax.scatter(Xp[rec_idx, 0], Xp[rec_idx, 1], color='red', s=80, label='Recommended')
        for i, name in zip(rec_idx, recs_df['ETF']):
            ax.text(Xp[i, 0]+0.05, Xp[i, 1]+0.05, name, fontsize=9, color='red')

        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend()
        ax.set_title("ETF Similarity Map (PCA 2D)")

        st.pyplot(fig)
