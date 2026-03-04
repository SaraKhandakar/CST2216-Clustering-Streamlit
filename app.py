import streamlit as st
import pandas as pd

from config import (
    DATA_PATH, LOG_PATH,
    FEATURE_SET_2D, FEATURE_SET_3D,
    K_MIN, K_MAX, DEFAULT_K
)
from src.utils import setup_logger
from src.data_loader import load_data
from src.clustering import run_kmeans, compute_wcss, compute_silhouette, cluster_summary
from src.visuals import plot_elbow, plot_silhouette, plot_clusters_2d, plot_clusters_pca

logger = setup_logger(LOG_PATH)

st.set_page_config(page_title="Customer Segmentation (KMeans)", layout="wide")
st.title("🛍️ Customer Segmentation — KMeans Clustering")
st.caption("Based on the Level-1 notebook: Elbow + Silhouette + KMeans clusters on Mall Customers dataset.")

@st.cache_data
def get_data():
    return load_data(DATA_PATH, logger)

df = get_data()

with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)

st.sidebar.header("⚙️ Settings")
feature_mode = st.sidebar.radio("Feature set", ["2D (Income + Spending)", "3D (Age + Income + Spending)"])

features = FEATURE_SET_2D if feature_mode.startswith("2D") else FEATURE_SET_3D
k = st.sidebar.slider("Number of clusters (k)", K_MIN, K_MAX, DEFAULT_K)

run_clicked = st.sidebar.button("Run Clustering")

# Always show k-selection helpers (matches notebook idea)
st.subheader("📈 Choosing k (Elbow + Silhouette)")
ks, wcss = compute_wcss(df, features, K_MIN, K_MAX, logger)
ks2, sil = compute_silhouette(df, features, K_MIN, K_MAX, logger)

c1, c2 = st.columns(2)
with c1:
    st.pyplot(plot_elbow(ks, wcss))
with c2:
    st.pyplot(plot_silhouette(ks2, sil))

st.divider()
st.subheader("✅ Clustering Results")

if run_clicked:
    model, labels, centers = run_kmeans(df, features, k, logger)
    summary = cluster_summary(df, labels, features)

    left, right = st.columns([1.2, 1])

    with left:
        st.write("**Cluster summary (means + count)**")
        st.dataframe(summary, use_container_width=True)

    with right:
        st.write("**Cluster visualization**")
        if features == FEATURE_SET_2D:
            st.pyplot(plot_clusters_2d(df, features[0], features[1], labels, centers))
        else:
            st.pyplot(plot_clusters_pca(df[features], labels))
else:
    st.info("Choose a feature set + k in the sidebar, then click **Run Clustering**.")