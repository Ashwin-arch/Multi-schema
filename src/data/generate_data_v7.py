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


def ensure_dimensions():
    """
    Ensure dimension tables exist in data/raw/.
    If they exist, reuse them without regeneration to preserve integrity.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    customers_path = os.path.join(RAW_DIR, "customers.csv")
    products_path = os.path.join(RAW_DIR, "products.csv")
    dates_path = os.path.join(RAW_DIR, "dates.csv")

    if (
        os.path.exists(customers_path)
        and os.path.exists(products_path)
        and os.path.exists(dates_path)
    ):
        # Load products for unit prices
        product_prices = {}
        with open(products_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                product_prices[int(row["product_id"])] = float(row["unit_price"])
        return product_prices

    print("Generating deterministic dimension tables (seed=42)...")
    random.seed(SEED)

    # 1. Customers
    regions = ["North", "South", "East", "West"]
    segments = ["Consumer", "Corporate", "Home Office"]
    with open(customers_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["customer_id", "customer_name", "region", "segment"])
        for cid in range(1, 1001):
            writer.writerow([
                cid,
                f"Customer_{cid}",
                random.choice(regions),
                random.choice(segments)
            ])

    # 2. Products
    categories = {
        "Electronics": ["Laptop", "Mobile", "Tablet", "Monitor"],
        "Furniture": ["Chair", "Table", "Desk", "Cabinet"],
        "Office": ["Printer", "Paper", "Pen", "File"]
    }
    product_prices = {}
    pid = 1
    with open(products_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["product_id", "product_name", "category", "unit_price"])
        for cat, names in categories.items():
            for name in names:
                for i in range(1, 6):
                    price = round(random.uniform(100, 5000), 2)
                    writer.writerow([pid, f"{name}_{i}", cat, price])
                    product_prices[pid] = price
                    pid += 1

    # 3. Dates
    start_date = datetime(2025, 1, 1)
    with open(dates_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "date_id", "full_date", "year", "quarter", "month", "month_name", "day"
        ])
        for i in range(365):
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

    print("Dimension tables initialized.")
    return product_prices


def generate_sales_data(rows=100_000, output_path=None, force=False):
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

    # If exact 100K and data/raw/sales.csv exists, verify if we can link/copy
    default_100k = os.path.join(RAW_DIR, "sales.csv")
    if rows == 100_000 and os.path.exists(default_100k) and output_path != default_100k:
        if not os.path.exists(output_path):
            import shutil
            shutil.copyfile(default_100k, output_path)

    # If target output already exists and has correct rows, reuse it
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            line_count = sum(1 for _ in f) - 1
        if line_count == rows:
            print(f"Dataset {output_path} already exists ({line_count:,} rows). Reusing.")
            return output_path

    product_prices = ensure_dimensions()

    print(f"\nGenerating {rows:,} sales rows for scale {scale_label} -> {output_path}...")
    start_time = time.perf_counter()

    # Re-align random seed state precisely
    random.seed(SEED)
    # Fast-forward RNG state past dimension table generation
    regions = ["North", "South", "East", "West"]
    segments = ["Consumer", "Corporate", "Home Office"]
    for _ in range(1000):
        random.choice(regions)
        random.choice(segments)
    for _ in range(60):
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
            customer_id = random.randint(1, 1000)
            product_id = random.randint(1, len(product_prices))
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
    if rows == 100_000 and not os.path.exists(default_100k):
        import shutil
        shutil.copyfile(output_path, default_100k)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Deterministic scalable sales fact generator.")
    parser.add_argument("--rows", type=int, default=100_000, help="Number of sales rows (e.g. 100000, 500000, 1000000)")
    parser.add_argument("--output", type=str, default=None, help="Target output file path")
    parser.add_argument("--force", action="store_true", help="Force generation bypassing resource check")
    args = parser.parse_args()

    generate_sales_data(rows=args.rows, output_path=args.output, force=args.force)


if __name__ == "__main__":
    main()
