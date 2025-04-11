# Organize Media AI

This Python script organizes images and videos by date and event, renames them systematically, detects duplicates (exact and near), and uses AI to label images.

## 📁 Features

- 🗃️ Organizes files into `/destination/YYYY-MM-DD/EventName`
- 🧠 Labels images using a pretrained ResNet-50 model
- 🔒 Detects exact duplicates using SHA-256
- 🧬 Detects near-duplicates (images & videos) using perceptual hashing
- 📝 Stores rename counters in `organize_config.json`
- 🎥 Uses multiple video frames to improve duplicate accuracy

## 🛠️ Requirements

```bash
pip install pillow torchvision timm exifread imagehash opencv-python pyheif
```
On macOS, you may also need:

```bash
brew install libheif
```
## 🚀 Usage
Put all your unsorted media in the /source folder.

Run the script:
```bash
python organize_media.py
```
Sorted files will be moved to /destination.

## 🧾 Output Structure

```plaintext
/destination/
├── 2024-03-10/
│   └── Birthday/
│       ├── pict00001_dog.jpg
│       └── vid00001_party.mp4
├── duplicated/
├── near_duplicated/
└── noevaluate/
```

📦 Configuration
The script stores its counters in organize_config.json, so it can resume where it left off on the next run.

---

## 🔧 Configuration File

After the first run, `organize_config.json` is created:

```json
{
  "pic_counter": 34,
  "vid_counter": 12
}
```

This ensures that counters continue across multiple runs.

---

## 🧠 AI Classification

Images are classified with `ResNet-50` using PyTorch + TIMM. The label is added to the filename automatically.

Example:
```
Original: IMG_1234.JPG
Renamed:  pict00012_golden_retriever.jpg
```

---

## 🧬 Duplicate Detection

- **Exact duplicates**: Detected with `SHA-256` file hash
- **Near-duplicates**:
  - Images: `imagehash.phash`
  - Videos: Sample 3 frames → compute average perceptual hash

These are moved to:
```
/destination/exact_duplicated/
/destination/near_duplicated/
```

---

## ✅ How to Use

1. Drop unsorted media into `/source`
2. Run the script:

```bash
python organize_media.py
```

3. Organized files will be moved to `/destination`

---

## 🧪 Tested On

- Python 3.10+
- macOS and Linux
- NVIDIA CUDA GPU (optional for faster AI classification)

---

## 📜 License

MIT License

---

## 🙌 Contributions

Feel free to fork and contribute! Ideas, PRs, and issues are welcome.

---

Happy organizing! 🧹📁📸

---

## 🧑‍💻 Author

Developed by [Nynor-code](https://github.com/Nynor-code).