'''
Organize media files from a source directory to a destination directory.
- Pictures are classified using a pre-trained model and renamed accordingly.
- Videos are processed to extract frames and generate perceptual hashes for near-duplicate detection.
- Files are organized into folders based on their date and event.
- Duplicates and near-duplicates are handled by copying them to separate folders.
'''
# Import necessary libraries
import os
import shutil
import exifread
import pyheif
import json
from PIL import Image
from datetime import datetime
import torch
import timm
import torchvision.transforms as transforms
from pathlib import Path
import hashlib
import imagehash
import cv2
import argparse
import urllib.request


# Setup paths
# SOURCE_DIR = "/Volumes/NFP4TBSSD/pyton_photos/Photos_Base"
# DEST_DIR = "/Volumes/NFP4TBSSD/pyton_photos/Photos_Base_Organized"
# CONFIG_FILE = "cfg/organize_config.json"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEST_MOMENT = False # True for test, False for full dataset - CHANGE MANUALLY

def parse_args():
    '''
    Parse command line arguments.
    Returns:
    argparse.Namespace: Parsed arguments including source, destination, config file, verbosity level, and near-duplicate threshold.
    '''
    # Create arguments parser
    parser = argparse.ArgumentParser(description="Organize media files.")
    
    # Local data Folders
    test = TEST_MOMENT  # True for test, False for full dataset - CHANGE MANUALLY
    if test:
        # Folders for the test dataset
        parser.add_argument("--source", type=str, default="/Volumes/NFP4TBSSD/pyton_photos/Photos_Base_Subset") # Local Filepath
        parser.add_argument("--dest", type=str, default="/Volumes/NFP4TBSSD/pyton_photos/Photos_Base_Subset_Organized") # Local Filepath
    else:
        # Folders for the full dataset
        parser.add_argument("--source", type=str, default="/Volumes/NFP4TBSSD/pyton_photos/Photos_Base_Test") # Local Filepath
        parser.add_argument("--dest", type=str, default="/Volumes/NFP4TBSSD/pyton_photos/Photos_Base_Organized_Test") # Local Filepath
        
    # configurations
    parser.add_argument("--config", type=str, default=os.path.join(ROOT_DIR, "cfg", "organize_config.json"), help="Path to the config file")
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity level (0 = no output, >0 = detailed output)")
    parser.add_argument("--near_duplicate_threshold", type=int, default=5, help="Threshold for near-duplicate detection")
    parser.add_argument("--processed_files_hash", type=str, default=os.path.join(ROOT_DIR, "cfg", "processed_files_hash.json"), help="Path to the processed files JSON file")
    parser.add_argument("--labels_path_url", type=str, default="https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", help="URL to download the labels file")
    parser.add_argument("--labels_path_local", type=str, default=os.path.join(ROOT_DIR, "data", "imagenet_classes.txt"), help="Local path to the labels file")
    
    return parser.parse_args()

def load_processed_hashes(processed_hashes_file):
    '''
    Load the processed file hashes from a JSON file.
    Parameters:
    processed_hashes_file (str): Path to the JSON file containing processed file hashes.
    Returns:
    set: Set of processed file hashes. If the file does not exist, return an empty set.
    '''
    if os.path.exists(processed_hashes_file):
        with open(processed_hashes_file, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_hashes(processed_hashes, processed_hashes_file):
    '''
    Save the processed file hashes to a JSON file.
    Parameters:
    processed_hashes (set): Set of processed file hashes to save.
    processed_hashes_file (str): Path to the JSON file to save the hashes.
    '''
    with open(processed_hashes_file, "w") as f:
        json.dump(list(processed_hashes), f, indent=2)

# Load or initialize configuration
def load_config(config_file):
    '''
    Load configuration from a JSON file. If the file does not exist, create a new one with default values.
    Returns:
    dict: Configuration dictionary with picture and video counters.
    '''
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return json.load(f)
    return {"pic_counter": 1, "vid_counter": 1}

def save_config(config, config_file):
    '''
    Save the configuration to a JSON file.
    Parameters:
    config (dict): Configuration dictionary to save.
    '''
    with open(config_file, "w") as f:
        json.dump(config, f)

def is_image(file):
    '''
    Check if the file is an image based on its extension.
    Parameters:
    file (str): File name to check.
    Returns:
    bool: True if the file is an image, False otherwise.
    '''
    return file.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".gif", ".bmp", ".heic", ".heif"))

def is_video(file):
    '''
    Check if the file is a video based on its extension.
    Parameters:
    file (str): File name to check.
    Returns:
    bool: True if the file is a video, False otherwise.
    '''
    return file.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".wmv"))

def load_model(labels_path_local, labels_path_url):
    '''
    Load a pre-trained image classification model and labels.
    Returns:
    tuple: A tuple containing the model and labels.
    '''

    model = timm.create_model("resnet50", pretrained=True)
    model.eval()
    
    # Move model to GPU if available (MPS for Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    print(f"Model moved to {device}.")

    if not os.path.exists(labels_path_local):
        urllib.request.urlretrieve(labels_path_url, labels_path_local)
        print(f"Downloaded labels to {labels_path_local}")
    
    with open(labels_path_local) as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels, device

def classify_image(file_path, model, labels, device):
    '''
    Classify an image using a pre-trained model.
    Parameters:
    file_path (str): Path to the image file.
    model: Pre-trained model for classification.
    labels (list): List of labels for classification.
    device: torch.device (cuda, mps, or cpu)
    Returns:
    str: The predicted label for the image.
    ''' 
    try:
        image = Image.open(file_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)  # Move input to device

        model.eval()  # 🧠 Just in case
        with torch.no_grad():
            outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        return labels[predicted.item()].replace(" ", "_")

    except Exception as e:
        print(f"Classification failed: {e}")
        return None

def extract_date(file_path):
    '''
    Extract the date from the file's metadata.
    Parameters:
    file_path (str): Path to the file.
    Returns:
    datetime: The extracted date, or None if not found.
    '''
    try:
        if file_path.lower().endswith(".heic", ".heif"):
            pyheif.read(file_path)
            return datetime.fromtimestamp(os.path.getmtime(file_path))
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, stop_tag="DateTimeOriginal", details=False)
            date_taken = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
            if date_taken:
                return datetime.strptime(str(date_taken), "%Y:%m:%d %H:%M:%S")
    except:
        pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(file_path))
    except:
        return None

def get_event_from_path(path):
    '''
    Extract the event name from the file path.
    Parameters:
    path (str): Path to the file.
    Returns:
    str: The extracted event name, or None if not found.
    '''
    parts = Path(path).parts
    for part in reversed(parts):
        if part.lower() not in ["source"]:
            return part.replace(" ", "_")
    return None

def generate_new_name(prefix, counter, label):
    '''
    Generate a new name for the file based on its type, counter, and label.
    Parameters:
    prefix (str): Prefix for the new name (e.g., "pict" or "vid").
    counter (int): Counter for the new name.
    label (str): Label for the new name.
    Returns:
    str: The generated new name.
    '''
    label_suffix = f"_{label}" if label else ""
    return f"{prefix}{str(counter).zfill(5)}{label_suffix}"

def file_hash(file_path):
    '''
    Generate a hash for the file using SHA-256.
    Parameters:
    file_path (str): Path to the file.
    Returns:
    str: The generated hash
    '''
    hash_func = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def perceptual_hash(file_path):
    '''
    Generate a perceptual hash for the image file.
    Parameters:
    file_path (str): Path to the image file.
    Returns:
    imagehash: The generated perceptual hash.
    '''
    try:
        image = Image.open(file_path).convert("RGB")
        return imagehash.phash(image)
    except:
        return None

def video_phash(file_path):
    '''
    Generate a perceptual hash for the video file by extracting frames.
    Parameters:
    file_path (str): Path to the video file.
    Returns:
    imagehash: The generated perceptual hash.
    '''
    try:
        cap = cv2.VideoCapture(file_path)
        hashes = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(3):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total_frames / 3))
            ret, frame = cap.read()
            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hashes.append(imagehash.phash(image))
        cap.release()
        return sum(hashes) / len(hashes) if hashes else None
    except:
        return None

def organize_media(verbose=2, dry_run=False):
    '''
    Organizes media files from SOURCE_DIR to DEST_DIR based on their type and metadata.
    - Pictures are classified using a pre-trained model and renamed accordingly.
    - Videos are processed to extract frames and generate perceptual hashes for near-duplicate detection.
    - Files are organized into folders based on their date and event.
    - Duplicates and near-duplicates are handled by copying them to separate folders.
    
    Parameters:
    verbose (int): Verbosity level for logging. 0 = no output, >0 = detailed output.
    '''
    # time it
    import time
    start_time = time.time()
    if verbose > 0:
        print("Starting media organization...")

    # parse arguments
    args = parse_args()
    NEAR_DUPLICATE_THRESHOLD = args.near_duplicate_threshold
    SOURCE_DIR = args.source
    DEST_DIR = args.dest
    CONFIG_FILE = args.config
    PROCESSED_HASHES_FILE = args.processed_files_hash
    LABELS_PATH_URL = args.labels_path_url
    LABELS_PATH_LOCAL = args.labels_path_local
    verbose = args.verbose
    
    # load list of processed files
    processed_hashes = load_processed_hashes(PROCESSED_HASHES_FILE)
    
    # load model
    model, labels, device = load_model(LABELS_PATH_LOCAL, LABELS_PATH_URL)
    config = load_config(CONFIG_FILE)
    
    # load config
    pic_counter, vid_counter = config.get("pic_counter", 1), config.get("vid_counter", 1)
    hashes = set()
    phashes = {}
    # count number of files processed
    # for final report
    count_files = {
    'files_count' : 0,
    'files_processed' : 0,
    'files_skipped' : 0,
    'files_duplicated' : 0,
    'files_unknown' : 0,
    'files_near_duplicated' : 0,
    'files_pict' : 0,
    'files_vid' : 0
    }

    for root, _, files in os.walk(SOURCE_DIR):
        # skip if the folder is empty
        if not files:
            if verbose > 0:
                print(f"Skipping empty folder: {root}")
            continue
    
        for file in files:
            count_files["files_count"] += 1
            
            file_path = os.path.join(root, file)
            if not os.path.isfile(file_path):
                continue
            if file.lower() in [".ds_store", "desktop.ini"]:
                count_files['files_skipped'] += 1
                count_files['files_unknown'] += 1
                continue
            try:
                fhash = file_hash(file_path)
                
                # Skip if file hash exists in processed hashes file
                if fhash in processed_hashes:
                    print(f"Skipping already processed: {file_path}")
                    count_files['files_skipped'] += 1
                    continue
                
                if fhash in hashes:
                    dup_dest = os.path.join(DEST_DIR, "duplicated")
                    os.makedirs(dup_dest, exist_ok=True)
                    if not dry_run:
                        shutil.copy(file_path, os.path.join(dup_dest, file))
                        print(f"Duplicated: {file_path} -> {dup_dest}/{file}")
                    else:
                        print(f"[Dry Run] Would copy: {file_path} -> {dup_dest}/{file}")
                    
                    count_files['files_duplicated'] += 1
                    count_files['files_skipped'] += 1
                    continue
                else:
                    hashes.add(fhash)

                phash = perceptual_hash(file_path) if is_image(file) else video_phash(file_path)
                if phash:
                    for existing, path in phashes.items():
                        if phash - existing <= NEAR_DUPLICATE_THRESHOLD:
                            near_dup_dest = os.path.join(DEST_DIR, "near_duplicated")
                            os.makedirs(near_dup_dest, exist_ok=True)
                            if not dry_run:
                                shutil.copy(file_path, os.path.join(near_dup_dest, file))
                                print(f"Near duplicate: {file_path} -> {near_dup_dest}/{file}")
                            else:
                                print(f"[Dry Run] Near duplicate: {file_path} -> {near_dup_dest}/{file}")
                                
                            count_files['files_near_duplicated'] += 1
                            count_files['files_skipped'] += 1
                            raise StopIteration
                    phashes[phash] = file_path

                date = extract_date(file_path)
                event = get_event_from_path(root)
                label = classify_image(file_path, model, labels, device) if is_image(file) else None

                if is_image(file):
                    prefix = "pict"
                    new_name = generate_new_name(prefix, pic_counter, label)
                    pic_counter += 1
                    #files_pict += 1
                    count_files['files_pict'] += 1
                    count_files['files_processed'] += 1
                elif is_video(file):
                    prefix = "vid"
                    new_name = generate_new_name(prefix, vid_counter, label)
                    vid_counter += 1
                    #files_vid += 1
                    count_files['files_vid'] += 1
                    count_files['files_processed'] += 1
                else:
                    dest = os.path.join(DEST_DIR, "toevaluate")
                    os.makedirs(dest, exist_ok=True)
                    if not dry_run:
                        shutil.copy(file_path, os.path.join(dest, file))
                        print(f"Unknown file type: {file_path} -> {dest}/{file}")
                    else:
                        print(f"[Dry Run] Unknown file type: {file_path} -> {dest}/{file}")

                    count_files['files_skipped'] += 1
                    continue

                ext = Path(file_path).suffix.lower()
                new_name += ext

                if date:
                    date_folder = date.strftime("%Y/%m-%d")
                    base_dest = os.path.join(DEST_DIR, date_folder)
                    if event and event.lower() != "source":
                        base_dest = os.path.join(base_dest, event)
                else:
                    base_dest = os.path.join(DEST_DIR, "nodate")

                os.makedirs(base_dest, exist_ok=True)
                if not dry_run:
                    shutil.copy(file_path, os.path.join(base_dest, new_name))
                    print(f"Copied: {file_path} -> {os.path.join(base_dest, new_name)}")
                else:
                    print(f"[Dry Run] Would copy: {file_path} -> {os.path.join(base_dest, new_name)}")
                    

            except StopIteration:
                continue
            except Exception as e:
                
                fallback_dest = os.path.join(DEST_DIR, "toevaluate")
                os.makedirs(fallback_dest, exist_ok=True)
                if not dry_run:
                    shutil.copy(file_path, os.path.join(fallback_dest, file))
                    print(f"Error processing {file_path}: {e}")
                else:
                    print(f"[Dry Run] Error processing {file_path}: {e}")

    # save processed hashes in hashes file
    save_processed_hashes(hashes, PROCESSED_HASHES_FILE)

    # save counters in config file
    config["pic_counter"] = pic_counter
    config["vid_counter"] = vid_counter
    save_config(config, CONFIG_FILE)
    
    # time it (END)
    end_time = time.time()
    duration = end_time - start_time
    # Print Summary Report
    print("\nSummary Report:")
    print(f"\nFinished in {duration/60:.2f} minutes.")
    print("Summary Report:")
    for k, v in count_files.items():
        print(f"  {k}: {v}")
    

if __name__ == '__main__':
    organize_media()
    print("Media organization completed.")