# Brain Tumour Detection and Classification - YOLOv11 (segmentation)
# Google Colab par run karo (Runtime > Change runtime type > T4 GPU)
# Har "# %%" ek Colab cell hai.

# %% Install
# !pip install -q ultralytics roboflow gradio opencv-python matplotlib

# %% Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# %% Imports
import os
import shutil
from getpass import getpass
from collections import defaultdict

import matplotlib.pyplot as plt
from roboflow import Roboflow
from ultralytics import YOLO

# %% Download dataset from Roboflow
# NOTE: API key code me hard-code mat karo. Roboflow > Settings > API Keys se nayi key lo.
api_key = getpass("Roboflow API key: ")
rf = Roboflow(api_key=api_key)
project = rf.workspace("deneme-tfvg6").project("tumorneuro")
version = project.version(1)
dataset = version.download("yolov11")   # folder: tumorneuro-1

print(os.listdir("tumorneuro-1"))       # train, valid, test, data.yaml

# %% Class distribution
class_names = ['Glioma', 'Meningioma', 'Pituitary tumor']
dataset_base = "tumorneuro-1"
splits = ['train', 'valid', 'test']

split_counts = {s: defaultdict(int) for s in splits}
for split in splits:
    label_dir = os.path.join(dataset_base, split, 'labels')
    if not os.path.exists(label_dir):
        print(f"Warning: {label_dir} not found, skipping")
        continue
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file)) as f:
                for line in f:
                    if line.strip():
                        split_counts[split][int(line.split()[0])] += 1

totals = [sum(split_counts[s][i] for s in splits) for i in range(len(class_names))]

plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, totals)
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., h, f'{int(h):,}', ha='center', va='bottom')
plt.title("Brain Tumor Class Distribution")
plt.xlabel("Tumor Type")
plt.ylabel("Number of Instances")
plt.grid(axis='y', alpha=0.3)
plt.show()

# %% Train/Val/Test stacked chart
split_colors = {'train': '#1f77b4', 'valid': '#ff7f0e', 'test': '#2ca02c'}
fig, ax = plt.subplots(figsize=(12, 7))
for i, name in enumerate(class_names):
    bottom = 0
    for split in splits:
        count = split_counts[split][i]
        ax.bar(name, count, bottom=bottom, color=split_colors[split],
               label=split if i == 0 else "")
        bottom += count
ax.set_title("Class Distribution with Train/Val/Test Split")
ax.set_xlabel("Tumor Type")
ax.set_ylabel("Number of Instances")
ax.legend(title="Split")
plt.show()

# %% Train
model = YOLO("yolo11n-seg.pt")
train_results = model.train(
    data="/content/tumorneuro-1/data.yaml",
    epochs=10,      # zyada epochs = better accuracy (e.g. 30-50)
    imgsz=640,
    batch=16,
    device=0,
)

# %% Training results (plots)
from IPython.display import Image, display
run_dir = str(model.trainer.save_dir)   # e.g. /content/runs/segment/train
for name in ["results.png", "confusion_matrix_normalized.png", "train_batch0.jpg",
             "MaskF1_curve.png", "MaskPR_curve.png"]:
    p = os.path.join(run_dir, name)
    if os.path.exists(p):
        display(Image(p))

# %% Test on unseen images
test_results = model.predict(
    source="tumorneuro-1/test/images/",
    conf=0.25,
    save=True, save_txt=True, save_conf=True,
    show=False, show_labels=True, show_conf=True,
    device=0,
)

# %% Export ONNX + save weights to Google Drive
model.export(format="onnx")

out_dir = "/content/drive/MyDrive/Brain Tumour Detection"
os.makedirs(out_dir, exist_ok=True)
shutil.copy(os.path.join(run_dir, "weights", "best.pt"), os.path.join(out_dir, "best.pt"))
shutil.copy(os.path.join(run_dir, "weights", "best.onnx"), os.path.join(out_dir, "best.onnx"))
print("Model saved to Google Drive:", out_dir)
