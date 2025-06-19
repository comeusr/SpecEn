import os
import re

def remove_tqdm_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    cleaned_lines = []

    for line in lines:
        if "Processing prompts:" in line:
            continue  # Remove any line that contains tqdm progress
        cleaned_lines.append(line)

    with open(filepath, 'w') as f:
        f.writelines(cleaned_lines)

def clean_all_logs(folder):
    for dirpath, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith(".log"):
                remove_tqdm_lines(os.path.join(dirpath, filename))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", help="Root folder containing .log files", type=str)
    args = parser.parse_args()

    clean_all_logs(args.log_folder)
