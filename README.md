# Retail Customer Behavior Segmentation

Identify distinct behavioral archetypes from retail transaction data to enhance targeted marketing strategies using unsupervised learning.

This project implements a full pipeline to transform raw transaction records into actionable customer segments using **RFM Analysis** and **K-Means Clustering**.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
Execute the full process (preprocessing, model training, and visualization):
```bash
PYTHONPATH=. python3 main.py
```

### 3. Explore Interactively
Open the Jupyter Notebook for detailed EDA and interactive 3D visualizations:
```bash
jupyter notebook notebooks/Retail_Segmentation.ipynb
```

## 🧠 How It Works

### 1. RFM Feature Engineering
The pipeline converts raw transactions into three core metrics:
*   **Recency (R)**: Days since the last purchase.
*   **Frequency (F)**: Total number of transactions.
*   **Monetary (M)**: Total spend per customer.

### 2. Skewness Correction
Real-world retail data is highly skewed (e.g., a few "whales" spend significantly more than others). To ensure the clustering algorithm isn't biased by extreme outliers, we apply a **Log Transformation** to the RFM features before scaling.

### 3. Unsupervised Clustering
*   **Algorithm**: K-Means Clustering.
*   **Scaling**: StandardScaler is applied to ensure all metrics have equal weight.
*   **Optimization**: The model uses the **Elbow Method** to confirm the optimal number of segments (k=4).

## 📊 Identified Segments

| Segment | Archetype | Strategy |
| :--- | :--- | :--- |
| **Champions** | High spend, frequent, recent. | Exclusive previews, loyalty rewards. |
| **Loyal Customers** | Steady spenders, regular visits. | Cross-sell and upsell high-margin products. |
| **At Risk** | Formerly valuable, but inactive. | Re-engagement campaigns, deep discounts. |
| **Churned/Low Value** | One-off or infrequent low spenders. | Automated low-cost reachouts. |

## 📂 Project Structure

```text
├── data/
│   ├── customer_segments.csv   # Final results with labels
│   └── segmentation_plot.png   # Cluster visualization
├── notebooks/
│   └── Retail_Segmentation.ipynb # Lab for detailed analysis
├── src/
│   ├── rfm_analysis.py         # Data cleaning & RFM logic
│   ├── clustering_model.py     # AI logic (K-Means & scaling)
│   └── data_generator.py       # Script for synthetic fallback
├── sample/
│   └── online_retail.csv       # Raw source dataset
├── main.py                     # Project entry point
└── requirements.txt            # Dependency list
```

## 🛠️ Tech Stack
*   **Engine**: Python 3.x
*   **Data**: Pandas, NumPy
*   **AI**: Scikit-Learn (K-Means)
*   **Viz**: Matplotlib, Seaborn

---
> [!TIP]
> To adjust the number of clusters, modify the `n_clusters` parameter in `main.py`.
