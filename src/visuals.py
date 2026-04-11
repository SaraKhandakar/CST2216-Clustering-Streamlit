# =========================
# Visualization Module
# =========================
# This file contains functions used to visualize clustering results,
# including elbow plot, silhouette plot, 2D cluster plots, and PCA projection.

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_elbow(ks, wcss):
    """
    Create an elbow plot for KMeans clustering.

    Parameters:
    ks (list): List of tested k values
    wcss (list): Corresponding WCSS values

    Returns:
    matplotlib.figure.Figure: Elbow plot figure

    Purpose:
    The elbow plot helps identify a suitable number of clusters
    by showing how WCSS changes as k increases.
    """
    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot k values against WCSS
    ax.plot(ks, wcss, marker="o")

    # Add labels and title
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("WCSS (Inertia)")
    ax.set_title("Elbow Plot")

    return fig


def plot_silhouette(ks, scores):
    """
    Create a silhouette score plot for KMeans clustering.

    Parameters:
    ks (list): List of tested k values
    scores (list): Corresponding silhouette scores

    Returns:
    matplotlib.figure.Figure: Silhouette plot figure

    Purpose:
    The silhouette plot helps evaluate how well-separated
    and compact the clusters are for different k values.
    """
    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot k values against silhouette scores
    ax.plot(ks, scores, marker="o")

    # Add labels and title
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Plot")

    # Silhouette scores range from -1 to 1
    ax.set_ylim(-1, 1)

    return fig


def plot_clusters_2d(df, x_col, y_col, labels, centers):
    """
    Plot KMeans clusters using two selected features.

    Parameters:
    df (DataFrame): Original dataset
    x_col (str): Feature for x-axis
    y_col (str): Feature for y-axis
    labels: Cluster labels assigned to each data point
    centers: Cluster centroids

    Returns:
    matplotlib.figure.Figure: 2D cluster scatter plot

    Purpose:
    This plot visualizes how data points are grouped into clusters
    using two selected features, along with the cluster centers.
    """
    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot clustered data points
    ax.scatter(df[x_col], df[y_col], c=labels)

    # Plot cluster centers
    ax.scatter(centers[:, 0], centers[:, 1], s=200, c="black", alpha=0.6)

    # Add labels and title
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("KMeans Clusters (2D)")

    return fig


def plot_clusters_pca(df_features, labels):
    """
    Plot cluster assignments using PCA for 2D projection.

    Parameters:
    df_features (DataFrame): Feature set used for clustering
    labels: Cluster labels assigned to each data point

    Returns:
    matplotlib.figure.Figure: PCA-based 2D cluster plot

    Purpose:
    PCA reduces higher-dimensional feature space into two principal
    components so clusters can be visualized in two dimensions.
    """
    # Reduce features to two principal components
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(df_features)

    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot transformed data points colored by cluster label
    ax.scatter(X2[:, 0], X2[:, 1], c=labels)

    # Add labels and title
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Clusters (PCA 2D Projection)")

    return fig