import pandas as pd
import numpy as np
from datetime import datetime

class RFMAnalysis:
    def __init__(self, data_path="data/retail_transactions.csv"):
        self.data_path = data_path
        self.df = None
        self.rfm_df = None

    def load_and_preprocess(self):
        """Load and clean the retail transaction data."""
        self.df = pd.read_csv(self.data_path, encoding='ISO-8859-1') # Handle possible encoding issues
        self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"])
        
        # Cleanup: Remove rows without CustomerID
        self.df = self.df.dropna(subset=["CustomerID"])
        
        # Cleanup: Filter out cancelled transactions (Quantity < 0)
        self.df = self.df[self.df["Quantity"] > 0]
        
        # Calculate Total Price
        self.df["TotalPrice"] = self.df["Quantity"] * self.df["UnitPrice"]
        
        # Ensure CustomerID is integer (or string without decimals)
        self.df["CustomerID"] = self.df["CustomerID"].astype(int).astype(str)
        
        print(f"Loaded and cleaned {len(self.df)} transactions.")
        return self.df

    def calculate_rfm(self, analysis_date=None):
        """Calculate Recency, Frequency, and Monetary scores per customer."""
        if self.df is None:
            self.load_and_preprocess()

        if analysis_date is None:
            analysis_date = self.df["InvoiceDate"].max() + pd.Timedelta(days=1)
        
        self.rfm_df = self.df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (analysis_date - x.max()).days,
            "InvoiceNo": "count",
            "TotalPrice": "sum"
        })
        
        self.rfm_df.rename(columns={
            "InvoiceDate": "Recency",
            "InvoiceNo": "Frequency",
            "TotalPrice": "Monetary"
        }, inplace=True)
        
        # Ensure all values are positive (just in case)
        self.rfm_df = self.rfm_df[(self.rfm_df > 0).all(axis=1)]
        
        print(f"Calculated RFM for {len(self.rfm_df)} customers.")
        return self.rfm_df

    def handle_skewness(self, rfm_df):
        """Applying log transformation to handle skewness in RFM features."""
        rfm_log = np.log1p(rfm_df)
        return rfm_log

if __name__ == "__main__":
    rfm = RFMAnalysis()
    rfm_df = rfm.calculate_rfm()
    print(rfm_df.head())
