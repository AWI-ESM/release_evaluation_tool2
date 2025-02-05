def update_status(status):
    """Update the status CSV file with the current script's status."""
    status_data = {}

    # Read existing CSV file
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            status_data = {row[0]: row[1] for row in reader}

    # Update status for the current script
    status_data[SCRIPT_NAME] = status

    # Write updated CSV file
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Script", "Last Execution Status"])
        for script, stat in status_data.items():
            writer.writerow([script, stat])
