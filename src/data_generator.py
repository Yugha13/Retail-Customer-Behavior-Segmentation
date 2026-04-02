import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_retail_data(num_customers=500, num_transactions=10000, output_path="data/retail_transactions.csv"):
    """
    Generates synthetic retail transaction data.
    """
    np.random.seed(42)
    
    # Customer IDs
    customer_ids = np.arange(1000, 1000 + num_customers)
    
    # Store types/segments for synthetic variety
    customer_profiles = {
        "VIP": {"freq": (10, 30), "monetary": (100, 500), "weight": 0.1},
        "Regular": {"freq": (3, 10), "monetary": (20, 100), "weight": 0.6},
        "Occasional": {"freq": (1, 3), "monetary": (10, 50), "weight": 0.3}
    }
    
    profiles = np.random.choice(list(customer_profiles.keys()), size=num_customers, 
                               p=[p["weight"] for p in customer_profiles.values()])
    
    transactions = []
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    for i, customer_id in enumerate(customer_ids):
        profile = profiles[i]
        num_tx = np.random.randint(*customer_profiles[profile]["freq"])
        
        for _ in range(num_tx):
            # Random date within the year
            days_offset = np.random.randint(0, (end_date - start_date).days)
            invoice_date = start_date + timedelta(days=days_offset)
            
            # Transaction details
            invoice_no = f"INV{np.random.randint(100000, 999999)}"
            stock_code = f"SKU{np.random.randint(1001, 2000)}"
            quantity = np.random.randint(1, 15)
            unit_price = np.round(np.random.uniform(*customer_profiles[profile]["monetary"]), 2)
            
            transactions.append({
                "InvoiceNo": invoice_no,
                "StockCode": stock_code,
                "Description": f"Product {stock_code}",
                "Quantity": quantity,
                "InvoiceDate": invoice_date,
                "UnitPrice": unit_price,
                "CustomerID": customer_id,
                "Country": "United Kingdom"
            })

    df = pd.DataFrame(transactions)
    df = df.sort_values("InvoiceDate")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions for {num_customers} customers.")
    return df

if __name__ == "__main__":
    generate_retail_data()
