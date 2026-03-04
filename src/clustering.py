import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans(df: pd.DataFrame, features: list[str], k: int, logger):
    X = df[features].copy()
    logger.info(f"Training KMeans with k={k} on features={features}")
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(X)
    centers = model.cluster_centers_
    return model, labels, centers

def compute_wcss(df: pd.DataFrame, features: list[str], k_min: int, k_max: int, logger):
    X = df[features].copy()
    ks = list(range(k_min, k_max + 1))
    wcss = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        wcss.append(float(km.inertia_))
    logger.info(f"Computed WCSS for k={ks}")
    return ks, wcss

def compute_silhouette(df: pd.DataFrame, features: list[str], k_min: int, k_max: int, logger):
    X = df[features].copy()
    ks = list(range(k_min, k_max + 1))
    scores = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        labels = km.labels_
        score = float(silhouette_score(X, labels))
        scores.append(score)
    logger.info(f"Computed silhouette for k={ks}")
    return ks, scores

def cluster_summary(df: pd.DataFrame, labels, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["Cluster"] = labels
    counts = out["Cluster"].value_counts().sort_index()
    centers = out.groupby("Cluster")[features].mean()
    summary = centers.copy()
    summary["Count"] = counts
    return summary.reset_index()