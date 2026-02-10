import os
from datetime import datetime

STATUS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "status")

def update_status(script_name, status):
    """Write a per-script status file. No locking needed since each script writes its own file."""
    os.makedirs(STATUS_DIR, exist_ok=True)
    status_file = os.path.join(STATUS_DIR, script_name.strip().replace(".py", ""))
    with open(status_file, "w") as f:
        f.write(f"{status.strip()} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def get_all_status():
    """Read all status files and return sorted dict."""
    results = {}
    if not os.path.exists(STATUS_DIR):
        return results
    for fname in sorted(os.listdir(STATUS_DIR)):
        fpath = os.path.join(STATUS_DIR, fname)
        if os.path.isfile(fpath):
            with open(fpath) as f:
                results[fname] = f.read().strip()
    return results


