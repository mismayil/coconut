import pandas as pd
import json
import os
from datasets import load_from_disk

def extract_answer(solution):
    # extract answer in the form \\boxed{2009}, \\boxed{\\frac{4}{35}}, $x\\equiv -36 \\equiv \\boxed{9} \\pmod {45}$
    import re
    # Find the start of \\boxed{
    match = re.search(r"\\boxed\{", solution)
    if not match:
        return ""
    
    # Count braces to find the matching closing brace (handles nested braces)
    start = match.end()
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == '{':
            depth += 1
        elif solution[i] == '}':
            depth -= 1
        i += 1
    
    if depth == 0:
        return solution[start:i-1]
    else:
        return ""
    
df = pd.read_csv("data/math/math_full_classification_results.csv")
dataset = load_from_disk("data/math/math_with_steps_and_flags")

cls_map = {}

for _, row in df.iterrows():
    cls_map[row["problem"]] = [int(x) for x in row["difficulty_levels"].split(";")]

# Remove samples that have more than 15 steps
dataset = dataset.filter(lambda x: len(x["steps"]) <= 15)

# Add new columns to be consistent with the other datasets
dataset = dataset.add_column("question", [item["problem"] for item in dataset])
dataset = dataset.add_column("answer", [extract_answer(item["solution"]) for item in dataset])
dataset = dataset.add_column("steps_difficulties", [cls_map.get(item["problem"], [1]*len(item["steps"])) for item in dataset])

# split into train, val and test
dataset = dataset.shuffle(seed=42)
train_size = int(0.95 * len(dataset))
val_size = int(0.01 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset = dataset.select(range(0, train_size))
val_dataset = dataset.select(range(train_size, train_size + val_size))
test_dataset = dataset.select(range(train_size + val_size, len(dataset)))

# Create directory if it doesn't exist
os.makedirs("data/math", exist_ok=True)

# save datasets as json files (array of objects)
with open("data/math/math_train.json", "w") as f:
    json.dump([item for item in train_dataset], f, indent=2)

with open("data/math/math_val.json", "w") as f:
    json.dump([item for item in val_dataset], f, indent=2)

with open("data/math/math_test.json", "w") as f:
    json.dump([item for item in test_dataset], f, indent=2)