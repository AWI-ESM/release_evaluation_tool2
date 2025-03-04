import sys
import os
import csv

CSV_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "status.csv")

def update_status(script_name, status):
    """Update the status CSV file with the current script's status."""
    status_data = {}

    # Ensure the logs directory exists
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    # Read existing CSV file
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            
            for row in reader:
                if len(row) >= 2:  # Ensure the row has at least two columns
                    script = row[0].strip()
                    stat = row[1].strip()
                    status_data[script] = stat

    # Update status for the current script
    status_data[script_name.strip()] = status.strip()

    # Write updated CSV file
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Script", "Last Execution Status"])
        for script, stat in sorted(status_data.items()):  # Sorted for consistency
            writer.writerow([script, stat])


