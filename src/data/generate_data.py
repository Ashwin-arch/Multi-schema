import os
import random
from datetime import datetime, timedelta

import pandas as pd


SEED = 42
random.seed(SEED)

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


def generate_data():
    # Customer dimension
    customers = []

    regions = ["North", "South", "East", "West"]
    segments = ["Consumer", "Corporate", "Home Office"]

    for customer_id in range(1, 1001):
        customers.append({
            "customer_id": customer_id,
            "customer_name": f"Customer_{customer_id}",
            "region": random.choice(regions),
            "segment": random.choice(segments)
        })

    customer_df = pd.DataFrame(customers)

    # Product dimension
    categories = {
        "Electronics": ["Laptop", "Mobile", "Tablet", "Monitor"],
        "Furniture": ["Chair", "Table", "Desk", "Cabinet"],
        "Office": ["Printer", "Paper", "Pen", "File"]
    }

    products = []
    product_id = 1

    for category, names in categories.items():
        for name in names:
            for i in range(1, 6):
                products.append({
                    "product_id": product_id,
                    "product_name": f"{name}_{i}",
                    "category": category,
                    "unit_price": round(random.uniform(100, 5000), 2)
                })
                product_id += 1

    product_df = pd.DataFrame(products)

    # Date dimension
    start_date = datetime(2025, 1, 1)

    dates = []

    for i in range(365):
        current = start_date + timedelta(days=i)

        dates.append({
            "date_id": i + 1,
            "full_date": current.date(),
            "year": current.year,
            "quarter": (current.month - 1) // 3 + 1,
            "month": current.month,
            "month_name": current.strftime("%B"),
            "day": current.day
        })

    date_df = pd.DataFrame(dates)

    # Sales fact table
    sales = []

    for sale_id in range(1, 100001):
        customer_id = random.randint(1, 1000)
        product_id = random.randint(1, len(product_df))
        date_id = random.randint(1, 365)

        quantity = random.randint(1, 10)

        unit_price = product_df.loc[
            product_df["product_id"] == product_id,
            "unit_price"
        ].iloc[0]

        discount = round(random.uniform(0, 0.25), 2)

        total_amount = quantity * unit_price * (1 - discount)

        sales.append({
            "sale_id": sale_id,
            "date_id": date_id,
            "customer_id": customer_id,
            "product_id": product_id,
            "quantity": quantity,
            "discount": discount,
            "total_amount": round(total_amount, 2)
        })

    sales_df = pd.DataFrame(sales)

    # Save
    customer_df.to_csv(f"{RAW_DIR}/customers.csv", index=False)
    product_df.to_csv(f"{RAW_DIR}/products.csv", index=False)
    date_df.to_csv(f"{RAW_DIR}/dates.csv", index=False)
    sales_df.to_csv(f"{RAW_DIR}/sales.csv", index=False)

    print("Dataset generation completed.")
    print()
    print(f"Customers : {len(customer_df):,}")
    print(f"Products  : {len(product_df):,}")
    print(f"Dates     : {len(date_df):,}")
    print(f"Sales     : {len(sales_df):,}")


if __name__ == "__main__":
    generate_data()
