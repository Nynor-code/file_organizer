import os
import shutil
import exifread
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
from collections import defaultdict
import subprocess

# Define paths
SOURCE_DIR = "/Volumes/NFP4TBSSD/PHOTOS_ORGANIZE"
DEST_DIR = "/Volumes/NFP4TBSSD/zz_organized_photos"

# Load a pre-trained image classification model (e.g., timm's resnet50)
model = timm.create_model("resnet50", pretrained=True)
model.eval()

# ImageNet labels for class names
LABELS_PATH = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
LABELS = [line.strip() for line in open("imagenet_classes.txt")] if os.path.exists("imagenet_classes.txt") else []

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_metadata(image_path):
    """Extract date taken from image metadata"""
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    
    date_tag = tags.get('EXIF DateTimeOriginal') or tags.get('Image DateTime')
    if date_tag:
        return datetime.strptime(str(date_tag), "%Y:%m:%d %H:%M:%S")
    return None

def get_video_metadata(video_path):
    """Extract creation date from video metadata using ffmpeg"""
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format_tags=creation_time", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.stdout:
            return datetime.strptime(result.stdout.strip(), "%Y-%m-%dT%H:%M:%S.%fZ")
    except Exception:
        pass
    return None

def classify_image(image_path):
    """Classify an image and return the top label."""
    if not LABELS:
        return "Unknown"
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return LABELS[predicted.item()]

def organize_files():
    """Recursively organize files from source to destination"""
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = file.lower().split('.')[-1]
            
            # Get metadata
            date_taken = None
            event_name = None
            if file_ext in ['jpg', 'jpeg', 'png', 'tiff', 'gif']:
                date_taken = get_image_metadata(file_path)
                event_name = classify_image(file_path) if not date_taken else None
            elif file_ext in ['mp4', 'mov', 'avi', 'mkv', 'wmv']:
                date_taken = get_video_metadata(file_path)
            
            # Fallback date if metadata is missing
            if not date_taken:
                date_taken = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            # Format destination path
            date_str = date_taken.strftime("%Y/%m-%d")
            dest_folder = os.path.join(DEST_DIR, date_str + ("_" + event_name if event_name else ""))
            os.makedirs(dest_folder, exist_ok=True)
            
            # Move file
            shutil.move(file_path, os.path.join(dest_folder, file))
            print(f"Moved {file_path} -> {dest_folder}")

if __name__ == "__main__":
    organize_files()
