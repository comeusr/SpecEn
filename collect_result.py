import os
import re
import ast


def clean_numpy_wrappers(s):
    # Replace np.float64(...) with just the float inside
    return re.sub(r'np\.float64\(([^)]+)\)', r'\1', s)

def flatten_dict(d, parent_key='', sep='.'):
    """Recursively flattens a nested dictionary using dot notation."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def parse_log_file(filepath):
    accepted_tokens = 0
    total_tokens = 0
    last_dict = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Extract token counts
    for line in lines:
        match = re.match(r"\[Accepted Num Tokens\]:\s+(\d+)\s+\[Total Num Tokens\]:\s+(\d+)", line)
        if match:
            accepted_tokens += int(match.group(1))
            total_tokens += int(match.group(2))

    # Try to find the last complete dictionary using brace counting
    brace_count = 0
    dict_lines = []
    in_dict = False

    for line in reversed(lines):
        stripped = line.strip()
        if not in_dict:
            if stripped.endswith("}"):
                dict_lines.insert(0, stripped)
                brace_count += stripped.count("}") - stripped.count("{")
                if brace_count == 0 and "{" in stripped:
                    break
                in_dict = True
        else:
            dict_lines.insert(0, stripped)
            brace_count += stripped.count("}") - stripped.count("{")
            if brace_count == 0:
                break

    # Parse the collected dictionary
    if dict_lines:
        try:
            raw_dict_str = "\n".join(dict_lines)
            raw_dict_str = clean_numpy_wrappers(raw_dict_str)
            last_dict = ast.literal_eval(raw_dict_str)
            last_dict = flatten_dict(last_dict)
        except Exception as e:
            print(f"Warning: Failed to parse nested dict in {filepath}: {e}")

    return accepted_tokens, total_tokens, last_dict



def parse_all_logs(root_folder):
    summary = []

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".log"):
                log_path = os.path.join(dirpath, filename)
                acc, total, metrics = parse_log_file(log_path)
                summary.append({
                    "file": log_path,
                    "accepted_tokens": acc,
                    "total_tokens": total,
                    "acceptance rate": acc/(total+1),
                    **metrics  # unpack the dictionary metrics
                })

    return summary

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder", type=str, help="Root folder to search for .log files")
    args = parser.parse_args()

    results = parse_all_logs(args.log_folder)

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    print(df)
    save_path = os.path.join(args.log_folder, "summary.csv")
    df.to_csv(save_path, index=False)