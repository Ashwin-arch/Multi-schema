import os
import shutil
import sys


def is_colab():
    """Detect if running inside Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ


def get_disk_space_gb(path="."):
    """Return total, used, free disk space in GB."""
    total, used, free = shutil.disk_usage(path)
    return {
        "total_gb": round(total / (1024 ** 3), 2),
        "used_gb": round(used / (1024 ** 3), 2),
        "free_gb": round(free / (1024 ** 3), 2),
    }


def get_memory_info_gb():
    """Return system memory info in GB with pure Python fallback."""
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    meminfo[key] = int(val)
            total = meminfo.get("MemTotal", 0) * 1024
            avail = meminfo.get("MemAvailable", meminfo.get("MemFree", 0)) * 1024
            return {
                "total_gb": round(total / (1024 ** 3), 2),
                "available_gb": round(avail / (1024 ** 3), 2),
            }
    except Exception:
        # Fallback using os.sysconf
        try:
            pagesize = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            total = pagesize * phys_pages
            avail = pagesize * avail_pages
            return {
                "total_gb": round(total / (1024 ** 3), 2),
                "available_gb": round(avail / (1024 ** 3), 2),
            }
        except Exception:
            return {"total_gb": 8.0, "available_gb": 4.0}


# Estimated requirements per scale (rows).
# disk_gb  = measured footprint of the raw sales CSV plus all three DuckDB files.
# ram_gb   = measured peak process RSS of the benchmark stage (interpreter + pandas + DuckDB).
# time_est_s = measured benchmark wall-clock for 600 queries x 3 schemas x 5 repetitions.
# Calibrated on the reference machine; see README_COLAB.md for the measurement table.
SCALE_ESTIMATES = {
    100_000: {"label": "100K", "disk_gb": 0.01, "ram_gb": 0.27, "time_est_s": 57},
    500_000: {"label": "500K", "disk_gb": 0.04, "ram_gb": 0.31, "time_est_s": 74},
    1_000_000: {"label": "1M", "disk_gb": 0.07, "ram_gb": 0.34, "time_est_s": 103},
    5_000_000: {"label": "5M", "disk_gb": 0.31, "ram_gb": 0.43, "time_est_s": 272},
    10_000_000: {"label": "10M", "disk_gb": 0.61, "ram_gb": 0.60, "time_est_s": 600},
}


def check_scale_feasibility(rows, safety_factor=1.5):
    """
    Check if a target dataset scale can safely run without OOM or disk exhaustion.
    Returns (is_feasible: bool, reason: str, details: dict)
    """
    disk = get_disk_space_gb()
    mem = get_memory_info_gb()

    est = SCALE_ESTIMATES.get(rows)
    if not est:
        est = {
            "label": f"{rows // 1000}K",
            "disk_gb": (rows / 10_000_000) * 3.6,
            "ram_gb": (rows / 10_000_000) * 3.5,
            "time_est_s": (rows / 10_000_000) * 700,
        }

    required_disk = est["disk_gb"] * safety_factor
    required_ram = est["ram_gb"]

    free_disk = disk["free_gb"]
    avail_ram = mem["available_gb"]

    details = {
        "scale": est["label"],
        "rows": rows,
        "required_disk_gb": round(required_disk, 2),
        "free_disk_gb": free_disk,
        "required_ram_gb": round(required_ram, 2),
        "available_ram_gb": avail_ram,
    }

    if free_disk < required_disk:
        reason = (
            f"Insufficient disk space for {est['label']}: "
            f"Need ~{required_disk:.2f} GB (with safety margin), but only {free_disk:.2f} GB available."
        )
        return False, reason, details

    if avail_ram < required_ram:
        reason = (
            f"Insufficient available RAM for {est['label']}: "
            f"Need ~{required_ram:.2f} GB, but only {avail_ram:.2f} GB available."
        )
        return False, reason, details

    return True, f"Scale {est['label']} is feasible to execute.", details


def print_system_summary():
    """Print clean summary of environment and resources."""
    colab = is_colab()
    disk = get_disk_space_gb()
    mem = get_memory_info_gb()

    print("=" * 60)
    print("SYSTEM & RESOURCE VERIFICATION")
    print("=" * 60)
    print(f"Environment : {'Google Colab' if colab else 'Standard Linux'}")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Total RAM   : {mem['total_gb']} GB (Available: {mem['available_gb']} GB)")
    print(f"Total Disk  : {disk['total_gb']} GB (Free: {disk['free_gb']} GB)")
    print("=" * 60)


if __name__ == "__main__":
    print_system_summary()
    for scale in [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]:
        feasible, msg, info = check_scale_feasibility(scale)
        status = "FEASIBLE" if feasible else "SKIPPED/UNSAFE"
        print(f"[{status:14s}] {info['scale']:5s} (rows: {scale:,}) - {msg}")
