import os
import shutil

desktop = "/mnt/c/Users/Windows/Desktop"

folders = {
    'Images': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg'],
    "Documents": ['doc', 'docx', 'pdf', 'txt', 'xls', 'xlsx', 'ppt', 'pptx', "csv"],
    "Videos" : ["mp4", "avi", "mov", "mkv"],
    "Music" : ["mp3", "wav", "ogg", "flac"],
    "Archives" : ["zip", "tar", "gz", "rar", "7z"],
}

for folder in folders.keys():
    folder_path = os.path.join(desktop, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

for file in os.listdir(desktop):
    file_path = os.path.join(desktop, file)

    if os.path.isdir(file_path):
        continue
    for folder, extensions in folders.items():
        if any(file.lower().endswith(ext) for ext in extensions):
            shutil.move(file_path, os.path.join(desktop, folder))
            print(f"Moved {file} to {folder}")
            break
print("Desktop Cleaned!!!!")
