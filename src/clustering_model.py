from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ClusteringModel:
    def __init__(self, rfm_df):
        self.rfm_df = rfm_df
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.kmeans = None
        self.labels = None

    def preprocess(self):
        """Standardizing RFM features."""
        self.X_scaled = self.scaler.fit_transform(self.rfm_df)
        return self.X_scaled

    def find_optimal_clusters(self, max_k=10):
        """Finding the optimal number of clusters using Elbow and Silhouette Scores."""
        if self.X_scaled is None:
            self.preprocess()

        wcss = []
        silhouette_scores = []
        
        K = range(2, max_k + 1)
        for k in K:
            km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
            km.fit(self.X_scaled)
            wcss.append(km.inertia_)
            silhouette_scores.append(silhouette_score(self.X_scaled, km.labels_))
        
        return K, wcss, silhouette_scores

    def fit(self, n_clusters=4):
        """Run K-Means with specified number of clusters."""
        if self.X_scaled is None:
            self.preprocess()

        self.kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(self.X_scaled)
        
        self.rfm_df["Cluster"] = self.labels
        return self.rfm_df

    def analyze_clusters(self):
        """Calculating average RFM values per cluster."""
        cluster_summary = self.rfm_df.groupby("Cluster").agg({
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean",
        }).reset_index()
        
        # Rank by Monetary value for consistent naming
        cluster_summary = cluster_summary.sort_values(by="Monetary", ascending=False).reset_index(drop=True)
        
        segment_names = ["Champions", "Loyal Customers", "At Risk", "Churned/Low Value"]
        mapping = {idx: segment_names[i] if i < len(segment_names) else f"Cluster {idx}" 
                   for i, idx in enumerate(cluster_summary["Cluster"])}
        
        self.rfm_df["Segment"] = self.rfm_df["Cluster"].map(mapping)
        return self.rfm_df["Segment"].value_counts()

if __name__ == "__main__":
    # Test with dummy data if needed
    pass
