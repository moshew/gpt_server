#!/usr/bin/env python3

import os

# מזהה ייחודי להפרדה בין קבצים (חייב להיות זהה לקוד שלך)
FILE_DELIMITER = "###FILE_PATH###:"

# התיקיה שממנה תרצה להתחיל (תיקיית הבסיס)
ROOT_DIR = "."

# שם קובץ הפלט
OUTPUT_FILE = "gpt.txt"

def collect_files(root_dir):
    file_list = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # לא לכלול תיקיות נסתרות
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for filename in filenames:
            if filename == OUTPUT_FILE or filename.startswith('.'):
                continue
            filepath = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(filepath, root_dir)
            file_list.append(rel_path)
    return file_list

def write_all_files_to_one(root_dir, output_file):
    files = collect_files(root_dir)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for rel_path in files:
            full_path = os.path.join(root_dir, rel_path)
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as in_f:
                content = in_f.read()
            out_f.write(f"{FILE_DELIMITER}{rel_path}\n")
            out_f.write(content)
            out_f.write("\n\n")
    print(f"Collected {len(files)} files into {output_file}")

if __name__ == "__main__":
    write_all_files_to_one(ROOT_DIR, OUTPUT_FILE)
