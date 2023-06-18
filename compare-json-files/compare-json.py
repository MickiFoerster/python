import json
import sys

def compare_json_files(file1_path, file2_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        json_data1 = json.load(file1)
        json_data2 = json.load(file2)
    
    differences = find_differences(json_data1, json_data2)
    
    if not differences:
        print("The JSON files are identical.")
    else:
        print("Differences found:")
        for difference in differences:
            print(difference)

def find_differences(data1, data2, path=""):
    differences = []

    if isinstance(data1, dict) and isinstance(data2, dict):
        for key in set(data1.keys()).union(data2.keys()):
            new_path = f"{path}.{key}" if path else key
            if key not in data1:
                differences.append(f"Key '{new_path}' is present only in the second JSON file.")
            elif key not in data2:
                differences.append(f"Key '{new_path}' is present only in the first JSON file.")
            else:
                nested_differences = find_differences(data1[key], data2[key], new_path)
                differences.extend(nested_differences)
    elif isinstance(data1, list) and isinstance(data2, list):
        if len(data1) != len(data2):
            differences.append(f"Array '{path}' has different lengths ({len(data1)} vs {len(data2)}).")
        else:
            for i, (item1, item2) in enumerate(zip(data1, data2)):
                new_path = f"{path}[{i}]"
                nested_differences = find_differences(item1, item2, new_path)
                differences.extend(nested_differences)
    elif data1 != data2:
        differences.append(f"Value at '{path}' is different: {data1} != {data2}")

    return differences

# Check if two command-line arguments are provided
if len(sys.argv) != 3:
    print("Usage: python script.py <file1> <file2>")
    sys.exit(1)

# Get the file paths from command-line arguments
file1_path = sys.argv[1]
file2_path = sys.argv[2]

compare_json_files(file1_path, file2_path)
