import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_elbow(ks, wcss):
    fig, ax = plt.subplots()
    ax.plot(ks, wcss, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("WCSS (Inertia)")
    ax.set_title("Elbow Plot")
    return fig

def plot_silhouette(ks, scores):
    fig, ax = plt.subplots()
    ax.plot(ks, scores, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Plot")
    ax.set_ylim(-1, 1)
    return fig

def plot_clusters_2d(df, x_col, y_col, labels, centers):
    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col], c=labels)
    ax.scatter(centers[:, 0], centers[:, 1], s=200, c="black", alpha=0.6)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("KMeans Clusters (2D)")
    return fig

def plot_clusters_pca(df_features, labels):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(df_features)
    fig, ax = plt.subplots()
    ax.scatter(X2[:, 0], X2[:, 1], c=labels)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Clusters (PCA 2D Projection)")
    return fig