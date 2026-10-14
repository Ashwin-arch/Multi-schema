import argparse
import csv
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.system_check import check_scale_feasibility
from src.utils.resource_logging import record_stage_resource

SEED = 42
RAW_DIR = "data/raw"


def get_scale_label(rows):
    """Return clean human-readable scale label, e.g. 100K, 500K, 1M, 5M, 10M."""
    if rows >= 1_000_000:
        val = rows / 1_000_000
        return f"{int(val)}M" if val.is_integer() else f"{val:.1f}M"
    val = rows / 1_000
    return f"{int(val)}K" if val.is_integer() else f"{val:.1f}K"


def ensure_dimensions(num_customers=1000, num_products=60, num_dates=365):
    """
    Ensure dimension tables exist in data/raw/.
    If they exist with correct sizes, reuse them without regeneration.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    customers_path = os.path.join(RAW_DIR, "customers.csv")
    products_path = os.path.join(RAW_DIR, "products.csv")
    dates_path = os.path.join(RAW_DIR, "dates.csv")

    # Check if existing files have correct sizes
    if (
        os.path.exists(customers_path)
        and os.path.exists(products_path)
        and os.path.exists(dates_path)
    ):
        with open(customers_path, "r") as f:
            customer_rows = sum(1 for _ in f) - 1
        with open(products_path, "r") as f:
            product_rows = sum(1 for _ in f) - 1
        with open(dates_path, "r") as f:
            date_rows = sum(1 for _ in f) - 1

        if customer_rows == num_customers and product_rows == num_products and date_rows == num_dates:
            # Load products for unit prices
            product_prices = {}
            with open(products_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    product_prices[int(row["product_id"])] = float(row["unit_price"])
            print(f"Dimension tables already exist with correct sizes (C={num_customers}, P={num_products}, D={num_dates}). Reusing.")
            return product_prices

    print(f"Generating deterministic dimension tables (C={num_customers}, P={num_products}, D={num_dates}, seed={SEED})...")
    random.seed(SEED)

    # 1. Customers
    regions = ["North", "South", "East", "West"]
    segments = ["Consumer", "Corporate", "Home Office"]
    with open(customers_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["customer_id", "customer_name", "region", "segment"])
        for cid in range(1, num_customers + 1):
            writer.writerow([
                cid,
                f"Customer_{cid}",
                random.choice(regions),
                random.choice(segments)
            ])

# 2. Products
    # Scale categories and products based on num_products
    if num_products <= 60:
        categories = {
            "Electronics": ["Laptop", "Mobile", "Tablet", "Monitor"],
            "Furniture": ["Chair", "Table", "Desk", "Cabinet"],
            "Office": ["Printer", "Paper", "Pen", "File"]
        }
    elif num_products <= 1000:
        categories = {
            "Electronics": [f"Electronics_{i}" for i in range(1, 71)],
            "Furniture": [f"Furniture_{i}" for i in range(1, 51)],
            "Office": [f"Office_{i}" for i in range(1, 36)],
            "Clothing": [f"Clothing_{i}" for i in range(1, 51)],
            "Sports": [f"Sports_{i}" for i in range(1, 36)],
        }
    else:
        categories = {f"Category_{i}": [f"Product_{j}" for j in range(1, 21)] for i in range(1, (num_products // 20) + 2)}

    product_prices = {}
    pid = 1
    with open(products_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["product_id", "product_name", "category", "unit_price"])
        for cat, names in categories.items():
            for name in names:
                for i in range(1, 6):
                    if pid > num_products:
                        break
                    price = round(random.uniform(100, 5000), 2)
                    writer.writerow([pid, f"{name}_{i}", cat, price])
                    product_prices[pid] = price
                    pid += 1
                if pid > num_products:
                    break
            if pid > num_products:
                break

    # 3. Dates
    start_date = datetime(2025, 1, 1)
    with open(dates_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "date_id", "full_date", "year", "quarter", "month", "month_name", "day"
        ])
        for i in range(num_dates):
            current = start_date + timedelta(days=i)
            writer.writerow([
                i + 1,
                current.date().isoformat(),
                current.year,
                (current.month - 1) // 3 + 1,
                current.month,
                current.strftime("%B"),
                current.day
            ])

    print(f"Dimension tables initialized: {num_customers:,} customers, {num_products:,} products, {num_dates:,} dates.")
    return product_prices


def generate_sales_data(rows=100_000, output_path=None, force=False,
                        num_customers=1000, num_products=60):
    """
    Generate deterministic scalable sales fact data using constant memory chunking.
    Preserves exact referential integrity and statistical distribution.
    """
    scale_label = get_scale_label(rows)

    # Resource check
    feasible, reason, details = check_scale_feasibility(rows)
    if not feasible and not force:
        print(f"\n[RESOURCE LIMITATION WARNING]: {reason}")
        print("Skipping generation to prevent system crash.")
        return None

    os.makedirs(RAW_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(RAW_DIR, f"sales_{scale_label.lower()}.csv")

    # If target output already exists and has correct rows, reuse it
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            line_count = sum(1 for _ in f) - 1
        if line_count == rows:
            print(f"Dataset {output_path} already exists ({line_count:,} rows). Reusing.")
            return output_path

    product_prices = ensure_dimensions(num_customers, num_products, 365)

    print(f"\nGenerating {rows:,} sales rows for scale {scale_label} -> {output_path}...")
    start_time = time.perf_counter()

    # Re-align random seed state precisely
    random.seed(SEED)
    # Fast-forward RNG state past dimension table generation
    regions = ["North", "South", "East", "West"]
    segments = ["Consumer", "Corporate", "Home Office"]
    for _ in range(num_customers):
        random.choice(regions)
        random.choice(segments)
    for _ in range(num_products):
        random.uniform(100, 5000)

    chunk_size = 50_000
    total_written = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "sale_id", "date_id", "customer_id", "product_id",
            "quantity", "discount", "total_amount"
        ])

        chunk = []
        for sale_id in range(1, rows + 1):
            customer_id = random.randint(1, num_customers)
            product_id = random.randint(1, num_products)
            date_id = random.randint(1, 365)
            quantity = random.randint(1, 10)
            unit_price = product_prices[product_id]
            discount = round(random.uniform(0, 0.25), 2)
            total_amount = round(quantity * unit_price * (1 - discount), 2)

            chunk.append([
                sale_id, date_id, customer_id, product_id,
                quantity, discount, total_amount
            ])

            if len(chunk) >= chunk_size:
                writer.writerows(chunk)
                total_written += len(chunk)
                chunk = []
                print(f"\r  Written: {total_written:,} / {rows:,} rows...", end="", flush=True)

        if chunk:
            writer.writerows(chunk)
            total_written += len(chunk)

    elapsed = time.perf_counter() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\nDone in {elapsed:.2f}s ({total_written:,} rows, {file_size_mb:.2f} MB).")

    record_stage_resource(
        rows,
        generation_time_s=elapsed,
        generation_size_mb=file_size_mb,
    )

    # If this is the active scale, update sales.csv
    default_100k = os.path.join(RAW_DIR, "sales.csv")
    if rows == 100_000 and not os.path.exists(default_100k):
        import shutil
        shutil.copyfile(output_path, default_100k)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Deterministic scalable sales fact generator with configurable dimensions.")
    parser.add_argument("--rows", type=int, default=100_000, help="Number of sales rows")
    parser.add_argument("--output", type=str, default=None, help="Target output file path")
    parser.add_argument("--force", action="store_true", help="Force generation bypassing resource check")
    parser.add_argument("--num-customers", type=int, default=1000, help="Number of customers")
    parser.add_argument("--num-products", type=int, default=60, help="Number of products")
    args = parser.parse_args()

    generate_sales_data(
        rows=args.rows,
        output_path=args.output,
        force=args.force,
        num_customers=args.num_customers,
        num_products=args.num_products
    )


if __name__ == "__main__":
    main()