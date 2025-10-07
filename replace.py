#!/usr/bin/env python3
import re
from pathlib import Path

# Directories to process
directories = ["test", "src", "include"]

# Pattern to match and replacement
pattern = re.compile(r"var_array_t<GRACE_NSPACEDIM>")
replacement = "var_array_t"

# Loop over directories recursively
for dir_name in directories:
    dir_path = Path(dir_name)
    if not dir_path.exists():
        print(f"Directory '{dir_name}' does not exist, skipping.")
        continue

    for file_path in dir_path.rglob("*.*"):  # process all files
        if file_path.is_file():
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()
            
            new_content = pattern.sub(replacement, content)
            
            if new_content != content:
                with file_path.open("w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"Updated {file_path}")
