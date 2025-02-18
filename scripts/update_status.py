import os
import csv
from filelock import FileLock

CSV_FILE = "status.csv"
LOCK_FILE = CSV_FILE + ".lock"

def update_status(script_name, status):
    """Update the status CSV file with the current script's status safely."""
    script_name = script_name.ljust(30)  # Ensure fixed-width formatting
    status_data = {}

    lock = FileLock(LOCK_FILE)

    with lock:  # Ensures only one process modifies the file at a time
        # Read existing CSV file
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                status_data = {row[1]: row[0] for row in reader}  # Ensure correct order

        # Update status for the current script
        status_data[script_name] = status

        # Write updated CSV file with correct column order
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Status", "Script"])  # Ensure correct header
            for script, stat in status_data.items():
                writer.writerow([stat, script])  # Status first, Script second

