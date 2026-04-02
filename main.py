import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_generator import generate_retail_data
from src.rfm_analysis import RFMAnalysis
from src.clustering_model import ClusteringModel

def run_segmentation_pipeline():
    """
    Executes the full Customer Behavior Segmentation pipeline.
    """
    print("Step 1: Loading real transaction data...")
    data_path = "sample/online_retail.csv"
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return
    
    print("\nStep 2: Performing RFM Analysis...")
    rfm = RFMAnalysis(data_path=data_path)
    rfm_df = rfm.calculate_rfm()
    
    # NEW: Log transform handles skewness (real data is highly skewed)
    print("Applying log transformation for better clustering...")
    rfm_transformed = rfm.handle_skewness(rfm_df)
    
    print("\nStep 3: Finding optimal clusters and training model...")
    # Using log-transformed data for building the model
    model = ClusteringModel(rfm_transformed)
    model.preprocess()
    
    # Run the model with k=4
    # Note: we update rfm_df but the labels are calculated from rfm_transformed
    rfm_transformed_with_labels = model.fit(n_clusters=4)
    
    # Map the labels to segment names
    segments_counts = model.analyze_clusters()
    
    # Re-assigning to original rfm_df for summary (keeping labels and segments)
    rfm_df["Cluster"] = rfm_transformed_with_labels["Cluster"]
    rfm_df["Segment"] = rfm_transformed_with_labels["Segment"]
    
    print("\nStep 4: Summary of results:")
    print(segments_counts)
    
    # Save the results
    rfm_df.to_csv("data/customer_segments.csv")
    print("\nResults saved to data/customer_segments.csv")
    
    # Create simple visualizations
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rfm_df, x="Recency", y="Monetary", hue="Segment", palette="viridis")
    plt.title("Customer Segments: Recency vs Monetary")
    plt.savefig("data/segmentation_plot.png")
    print("Plot saved to data/segmentation_plot.png")

if __name__ == "__main__":
    run_segmentation_pipeline()
