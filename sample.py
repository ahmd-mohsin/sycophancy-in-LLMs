import json
import os

# Change this to your folder path if needed
filename = "./data_generation/sycophancy_eval_responses.json"

if filename.lower().endswith(".json"):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"From file: {filename}")
    # If the JSON is a list of entries
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and "problem_id" in entry:
                print(entry["problem_id"])
    # If the JSON is a dict with entries under some key, adjust here:
    elif isinstance(data, dict):
        # Example: entries under "items" key
        for entry in data.get("items", []):
            if isinstance(entry, dict) and "problem_id" in entry:
                print(entry["problem_id"])
