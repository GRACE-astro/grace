import os
import sys
import glob 

def rename_files_in_directory(directory, search_word, replacement_word, exclude_dirs):
    for root, dirs, files in os.walk(directory):
        # Modify dirs in-place to exclude certain directories
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in exclude_dirs]
        for file in files:
            if search_word in file:
                new_file_name = file.replace(search_word, replacement_word)
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, new_file_name)
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed: {old_file_path} -> {new_file_path}")
                except Exception as e:
                    print(f"Error renaming file {old_file_path}: {e}")

if __name__ == "__main__":
    dirs = ["test","include/grace","src","./"]
    search_words = ["thunder","THUNDER","Thunder"]
    replace_words = ["grace","GRACE","GRACE"]
    exclude_dirs = glob.glob("./build*")
    exclude_dirs += glob.glob("./.git")
    exclude_dirs += glob.glob("./doc")
    for directory in dirs:
        for i in range(len(replace_words)):
            rename_files_in_directory(directory,  search_words[i], replace_words[i], exclude_dirs)
    print("File renaming complete.")