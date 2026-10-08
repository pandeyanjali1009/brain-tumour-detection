# Brain Tumour Detection and Classification (YOLOv11-seg)

Glioma / Meningioma / Pituitary tumour detection + segmentation from MRI images.
Dataset: Roboflow `deneme-tfvg6/tumorneuro` (version 1). UI: Gradio.

## Files
- `train.py` - Colab cells: dataset download, class plots, training, testing, ONNX export, Drive save
- `app.py` - Gradio web app (upload MRI -> annotated image + JSON details)
- `requirements.txt`

## Steps (Google Colab, GPU on)
1. Colab me naya notebook kholo, Runtime > T4 GPU.
2. `train.py` ke cells ek-ek karke paste/run karo (pehli line `!pip install ...` uncomment karo).
3. Training ke baad `runs/segment/train/weights/best.pt` Drive me save ho jayega.
4. `app.py` ko same notebook me run karo (ya `best.pt` ko upload karke `WEIGHTS=best.pt python app.py`).

## Model settings
yolo11n-seg.pt, epochs=10, imgsz=640, batch=16, conf=0.25

## Note
Roboflow API key kabhi code/GitHub me mat daalo; `getpass` se enter karo.
