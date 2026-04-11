# =========================
# Clustering Module
# =========================
# This file contains functions for running KMeans clustering,
# evaluating cluster quality, and summarizing cluster results.

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_kmeans(df: pd.DataFrame, features: list[str], k: int, logger):
    """
    Train a KMeans clustering model on selected features.

    Parameters:
    df (pd.DataFrame): Input dataset
    features (list[str]): List of features used for clustering
    k (int): Number of clusters
    logger: Logger object for tracking execution

    Returns:
    tuple:
        model: Trained KMeans model
        labels: Cluster labels assigned to each data point
        centers: Coordinates of cluster centroids

    Purpose:
    This function applies KMeans clustering to group similar
    data points into k clusters based on the selected features.
    """

    # Select only the features used for clustering
    X = df[features].copy()

    # Log model training information
    logger.info(f"Training KMeans with k={k} on features={features}")

    # Initialize and train the KMeans model
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")

    # Fit the model and assign cluster labels
    labels = model.fit_predict(X)

    # Extract cluster centroids
    centers = model.cluster_centers_

    return model, labels, centers


def compute_wcss(df: pd.DataFrame, features: list[str], k_min: int, k_max: int, logger):
    """
    Compute Within-Cluster Sum of Squares (WCSS) for a range of k values.

    Parameters:
    df (pd.DataFrame): Input dataset
    features (list[str]): Features used for clustering
    k_min (int): Minimum number of clusters
    k_max (int): Maximum number of clusters
    logger: Logger object

    Returns:
    tuple:
        ks (list): List of k values tested
        wcss (list): Corresponding WCSS values

    Purpose:
    WCSS is used in the elbow method to help determine the
    optimal number of clusters. Lower WCSS means tighter clusters.
    """

    # Select clustering features
    X = df[features].copy()

    # Create list of k values to test
    ks = list(range(k_min, k_max + 1))
    wcss = []

    # Train KMeans for each k and store inertia value
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        wcss.append(float(km.inertia_))

    # Log WCSS computation
    logger.info(f"Computed WCSS for k={ks}")

    return ks, wcss


def compute_silhouette(df: pd.DataFrame, features: list[str], k_min: int, k_max: int, logger):
    """
    Compute silhouette scores for a range of k values.

    Parameters:
    df (pd.DataFrame): Input dataset
    features (list[str]): Features used for clustering
    k_min (int): Minimum number of clusters
    k_max (int): Maximum number of clusters
    logger: Logger object

    Returns:
    tuple:
        ks (list): List of k values tested
        scores (list): Corresponding silhouette scores

    Purpose:
    Silhouette score measures how well-separated the clusters are.
    A higher silhouette score generally indicates better clustering quality.
    """

    # Select clustering features
    X = df[features].copy()

    # Create list of k values to test
    ks = list(range(k_min, k_max + 1))
    scores = []

    # Train KMeans for each k and calculate silhouette score
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        labels = km.labels_
        score = float(silhouette_score(X, labels))
        scores.append(score)

    # Log silhouette computation
    logger.info(f"Computed silhouette for k={ks}")

    return ks, scores


def cluster_summary(df: pd.DataFrame, labels, features: list[str]) -> pd.DataFrame:
    """
    Create a summary table for the generated clusters.

    Parameters:
    df (pd.DataFrame): Original dataset
    labels: Cluster labels assigned to each row
    features (list[str]): Features used in clustering

    Returns:
    pd.DataFrame: Summary table containing cluster means and counts

    Purpose:
    This function helps interpret clustering results by showing
    the average feature values and number of observations in each cluster.
    """

    # Copy original dataset to avoid modifying it directly
    out = df.copy()

    # Add cluster labels to the dataset
    out["Cluster"] = labels

    # Count number of records in each cluster
    counts = out["Cluster"].value_counts().sort_index()

    # Calculate mean feature values for each cluster
    centers = out.groupby("Cluster")[features].mean()

    # Combine cluster centers with counts
    summary = centers.copy()
    summary["Count"] = counts

    # Return summary as a DataFrame
    return summary.reset_index()