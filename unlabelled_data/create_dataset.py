import json
import os

# Directory containing text files
input_dir = "nursing-shift-handover"
output_file = "output.jsonl"

# Get all text files in the directory
text_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".txt")]

with open(output_file, "w", encoding="utf-8") as out_f:
    for i, filename in enumerate(sorted(text_files), start=1):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        record = {"text": content, "meta": {"dataset": "tpch", "id": i}}
        json_line = json.dumps(record, ensure_ascii=False)
        out_f.write(json_line + "\n")

print(f"✅ Created {output_file} with {len(text_files)} entries.")
