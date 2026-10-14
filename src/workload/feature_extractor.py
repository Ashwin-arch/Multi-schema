import json
import re
import os

import pandas as pd


WORKLOAD_FILE = "data/processed/workload.json"
OUTPUT_FILE = "data/processed/workload_features.csv"


def count_pattern(query, pattern):
    return len(
        re.findall(
            pattern,
            query,
            flags=re.IGNORECASE
        )
    )


def extract_features():

    with open(WORKLOAD_FILE, "r") as file:
        workloads = json.load(file)

    records = []

    for workload_name, queries in workloads.items():

        for query_info in queries:

            query = query_info["sql"]

            join_count = count_pattern(
                query,
                r"\bJOIN\b"
            )

            group_by_present = int(
                bool(
                    re.search(
                        r"\bGROUP\s+BY\b",
                        query,
                        flags=re.IGNORECASE
                    )
                )
            )

            aggregation_present = int(
                bool(
                    re.search(
                        r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\(",
                        query,
                        flags=re.IGNORECASE
                    )
                )
            )

            filter_present = int(
                bool(
                    re.search(
                        r"\bWHERE\b",
                        query,
                        flags=re.IGNORECASE
                    )
                )
            )

            tables = re.findall(
                r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
                query,
                flags=re.IGNORECASE
            )

            records.append({
                "query_id": query_info["query_id"],
                "workload": workload_name,
                "query_type": query_info["type"],
                "join_count": join_count,
                "aggregation": aggregation_present,
                "filter": filter_present,
                "group_by": group_by_present,
                "tables_used": len(set(tables))
            })

    df = pd.DataFrame(records)

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Feature extraction completed.")
    print()
    print(df.to_string(index=False))
    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_features()
